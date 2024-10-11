#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoders import EncoderMLP, DecoderMLP
from predictors import Predictor
from priors import NormalPrior
from utils import MMD, compute_mmd_loss, top_k_indices_column
import csv


# Default parameters for the IdealPointNN model
encoders = {
    "text": {
        "encoder_input": "bow",
        "encoder_hidden_layers":[256],
        "encoder_non_linear_activation":"relu",
        "encoder_bias":True
    },
    "vote": {
        "encoder_hidden_layers":[256],
        "encoder_non_linear_activation":"relu",
        "encoder_bias":True
    },
    "discrete_choice": {
        "encoder_hidden_layers":[256],
        "encoder_non_linear_activation":None,
        "encoder_bias":True
    }
}

decoders = {
    "text": {
        "decoder_input": "bow",
        "decoder_hidden_layers":[256],
        "decoder_non_linear_activation":"relu",
        "decoder_bias":True
    },
    "vote": {
        "decoder_hidden_layers":[256],
        "decoder_non_linear_activation":"relu",
        "decoder_bias":True
    },
    "discrete_choice": {
        "decoder_hidden_layers":[],
        "decoder_non_linear_activation":None,
        "decoder_bias":True
    }
}

predictor = {
    "predictor_type": "classifier",
    "predictor_hidden_layers":[],
    "predictor_non_linear_activation":None,
    "predictor_bias":True
}

class IdealPointNN:
    """
    Wrapper class for the Neural Ideal Point Model.
    """

    def __init__(
        self,
        train_datasets=None,
        test_datasets=None,
        n_dims=1,
        update_prior=False,
        tol=0.001,
        encoders=encoders,
        encoder_include_ideology_covariates=True,
        decoders=decoders,
        predictor=predictor,
        num_epochs=1000,
        num_workers=4,
        batch_size=64,
        learning_rate=1e-3,
        dropout=0.2,
        print_every_n_epochs=1,
        print_every_n_batches=1000,
        print_dims=True,
        print_content_covariates=True,
        log_every_n_epochs=1000,
        patience=1,
        delta=0,
        w_prior=None,
        w_pred_loss=1,
        ckpt_folder="../ckpt",
        model_name="model",
        ckpt=None,
        device=None,
        seed=42,
        verbose=False,
        eval_mode=False,
        ignore_saves_loads=False,
    ):
        self.verbose = verbose
        self.eval = eval_mode
        self.model_name = model_name
        self.train_datasets = train_datasets
        self.device = device
        self.ignore_saves_loads = ignore_saves_loads

        # ici le modele check si il est en entrainement ou eval.
        if ckpt:
            print('Loading model from checkpoint: ', ckpt)

            if device == 'cpu':
                self.load_model(ckpt, map_location=torch.device('cpu'))
            
            else:
                self.load_model(ckpt) 
            print('Model loaded successfully.')
        else:
            train_data = train_datasets[0]

            self.n_dims = n_dims
            self.dim_labels = ["Dim_{}".format(i) for i in range(n_dims)]
            self.update_prior = update_prior
            self.tol = tol

            self.encoder_params = encoders
            self.decoder_params = decoders
            self.predictor_params = predictor

            self.encoder_include_ideology_covariates = encoder_include_ideology_covariates

            self.predictor_type = self.predictor_params['predictor_type']
            self.predictor_hidden_layers = self.predictor_params['predictor_hidden_layers']
            self.predictor_non_linear_activation = self.predictor_params['predictor_non_linear_activation']
            self.predictor_bias = self.predictor_params['predictor_bias']

            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            self.num_workers = num_workers
            self.dropout = dropout

            self.print_every_n_epochs = print_every_n_epochs
            self.print_every_n_batches = print_every_n_batches
            self.print_dims = print_dims
            self.print_content_covariates = print_content_covariates
            self.log_every_n_epochs = log_every_n_epochs
            self.patience = patience
            self.delta = delta
            self.w_prior = w_prior
            self.w_pred_loss = w_pred_loss
            self.ckpt_folder = ckpt_folder

            if not os.path.exists(ckpt_folder):
                os.makedirs(ckpt_folder)

            self.seed = seed
            if seed is not None:
                torch.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                np.random.seed(seed)

            if device is None:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")

            self.modalities = train_data.modalities
            self.ref_modality = self.modalities[0]

            if train_data.ideology is not None:
                ideology_covariate_size = train_data.M_ideology_covariates.shape[1]
                self.ideology_colnames = train_data.ideology_colnames
            else:
                ideology_covariate_size = 0

            self.content_covariate_size = {}
            for modality in self.modalities:
                blerg = train_data.data[modality].get('M_content_covariates', None)
                if blerg is not None:
                    content_covariate_size = blerg.shape[1]
                    self.content_colnames = train_data.data[modality].get('content_colnames', None)
                else:
                    content_covariate_size = 0
                self.content_covariate_size[modality] = content_covariate_size

            if train_data.prediction is not None:
                prediction_covariate_size = train_data.M_prediction_covariates.shape[1]
                self.prediction_colnames = train_data.prediction_colnames
            else:
                prediction_covariate_size = 0

            if train_data.labels is not None:
                labels_size = train_data.M_labels.shape[1]
                if self.predictor_type == "classifier":
                    n_labels = len(np.unique(train_data.M_labels))
                else:
                    n_labels = 1
            else:
                labels_size = 0

            self.Encoders = {}
            self.Decoders = {}

            for mod in self.modalities:

                encoder_hidden_layers = self.encoder_params[mod]['encoder_hidden_layers']
                encoder_non_linear_activation = self.encoder_params[mod]['encoder_non_linear_activation']
                encoder_bias = self.encoder_params[mod]['encoder_bias']

                decoder_hidden_layers = self.decoder_params[mod]['decoder_hidden_layers']
                decoder_non_linear_activation = self.decoder_params[mod]['decoder_non_linear_activation']
                decoder_bias = self.decoder_params[mod]['decoder_bias']

                content_covariate_size = self.content_covariate_size[mod]

                if mod =='text':
                    features_size = train_data.data[mod]['M_features'].shape[1]
                    self.bow_size = features_size
                    encoder_input = self.encoder_params[mod]['encoder_input']

                    if encoder_input == "bow":
                        self.input_size = features_size
                    else:
                        input_embeddings_size = train_data.data[mod]['M_embeddings'].shape[1]
                        self.input_size = input_embeddings_size                   
                    
                    self.id2token = train_data.data[mod]['id2token']

                    self.input_size = features_size

                    if self.encoder_include_ideology_covariates:
                        self.input_size += ideology_covariate_size

                    self.ideology_covariate_size = ideology_covariate_size
                    self.labels_size = labels_size

                    encoder_dims = [self.input_size]
                    encoder_dims.extend(encoder_hidden_layers)
                    encoder_dims.extend([n_dims])

                    self.Encoders[mod] = EncoderMLP(
                        encoder_dims=encoder_dims,
                        encoder_non_linear_activation=encoder_non_linear_activation,
                        encoder_bias=encoder_bias,
                        dropout=dropout,
                    ).to(self.device)

                    decoder_dims = [n_dims + content_covariate_size]
                    decoder_dims.extend(decoder_hidden_layers)
                    decoder_dims.extend([features_size])

                    self.Decoders[mod] = DecoderMLP(
                        decoder_dims=decoder_dims,
                        decoder_non_linear_activation=decoder_non_linear_activation,
                        decoder_bias=decoder_bias,
                        dropout=dropout,
                    ).to(self.device)

                elif mod == 'vote':

                    self.Encoders[mod] = {}
                    self.Decoders[mod] = {}

                    for train_data in train_datasets:

                        t = train_data.t

                        features_size = train_data.data[mod]['M_features'].shape[1]
                        self.bow_size = features_size
                        self.input_size = features_size

                        if self.encoder_include_ideology_covariates:
                            self.input_size += ideology_covariate_size

                        self.ideology_covariate_size = ideology_covariate_size
                        self.labels_size = labels_size

                        if self.verbose:
                            print('=============================')
                            print('t: ', t)
                            print('features_size: ', features_size)
                            print('ideology_covariate_size: ', ideology_covariate_size)
                            print('labels_size: ', labels_size)
                            print('input_size: ', self.input_size)
                        
                        encoder_dims = [self.input_size]
                        encoder_dims.extend(encoder_hidden_layers)
                        encoder_dims.extend([n_dims])

                        self.Encoders[mod][t] = EncoderMLP(
                            encoder_dims=encoder_dims,
                            encoder_non_linear_activation=encoder_non_linear_activation,
                            encoder_bias=encoder_bias,
                            dropout=dropout,
                        ).to(self.device)

                        d = train_data.data[mod]
                        decoder_dims = [n_dims + content_covariate_size]
                        decoder_dims.extend(decoder_hidden_layers)
                        decoder_dims.extend([d['M_features'].shape[1]])

                        self.Decoders[mod][t] = DecoderMLP(
                            decoder_dims=decoder_dims,
                            decoder_non_linear_activation=decoder_non_linear_activation,
                            decoder_bias=decoder_bias,
                            dropout=dropout,
                        ).to(self.device)

                elif mod == 'discrete_choice':    

                    self.Encoders[mod] = {}
                    self.Decoders[mod] = {}

                    for train_data in train_datasets:

                        t = train_data.t

                        features_size = train_data.data[mod]['M_features'].shape[1]
                        self.bow_size = features_size
                        self.input_size = features_size

                        if self.encoder_include_ideology_covariates:
                            self.input_size += ideology_covariate_size

                        self.ideology_covariate_size = ideology_covariate_size
                        self.labels_size = labels_size

                        encoder_dims = [self.input_size]
                        encoder_dims.extend(encoder_hidden_layers)
                        encoder_dims.extend([n_dims])

                        self.Encoders[mod][t] = EncoderMLP(
                            encoder_dims=encoder_dims,
                            encoder_non_linear_activation=encoder_non_linear_activation,
                            encoder_bias=encoder_bias,
                            dropout=dropout,
                        ).to(self.device)

                        d = train_data.data[mod]
                        self.Decoders[mod][t] = {}
                        for k,v in d.items(): # loop over variables
                            if k not in ["M_features", "M_colnames"]:
                                decoder_dims = [n_dims + content_covariate_size]
                                decoder_dims.extend(decoder_hidden_layers)
                                decoder_dims.extend([v['M_features'].shape[1]])

                                self.Decoders[mod][t][k] = DecoderMLP(
                                    decoder_dims=decoder_dims,
                                    decoder_non_linear_activation=decoder_non_linear_activation,
                                    decoder_bias=decoder_bias,
                                    dropout=dropout,
                                ).to(self.device)

            self.prior = NormalPrior(
                ideology_covariate_size,
                n_dims,
                device=self.device,
            )

            if labels_size != 0:
                predictor_dims = [n_dims + prediction_covariate_size]
                predictor_dims.extend(self.predictor_hidden_layers)
                predictor_dims.extend([n_labels])
                self.predictor = Predictor(
                    predictor_dims=predictor_dims,
                    predictor_non_linear_activation=self.predictor_non_linear_activation,
                    predictor_bias=self.predictor_bias,
                    dropout=dropout,
                ).to(self.device)

            list_of_encoder_parameters = []
            list_of_decoder_parameters = []
            for modality in self.modalities:
                if modality == 'text':
                    list_of_encoder_parameters += list(self.Encoders[modality].parameters())
                    list_of_decoder_parameters += list(self.Decoders[modality].parameters())
                elif modality == 'vote':
                    for train_data in train_datasets:
                        t = train_data.t
                        list_of_encoder_parameters += list(self.Encoders[modality][t].parameters())
                        list_of_decoder_parameters += list(self.Decoders[modality][t].parameters())
                elif modality == 'discrete_choice':
                    for train_data in train_datasets:
                        t = train_data.t
                        list_of_encoder_parameters += list(self.Encoders[modality][t].parameters())
                        for k,v in self.Decoders[modality][t].items():
                            list_of_decoder_parameters += list(self.Decoders[modality][t][k].parameters())

            if self.labels_size != 0: 
                list_of_predictor_parameters = list(self.predictor.parameters())
                self.optimizer = torch.optim.Adam(
                    list_of_encoder_parameters + list_of_decoder_parameters + list_of_predictor_parameters,
                    lr=self.learning_rate,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    list_of_encoder_parameters + list_of_decoder_parameters, 
                    lr=self.learning_rate
                )

            self.epochs = 0
            self.reconstruction_loss = None
            self.mmd_loss = None
            self.prediction_loss = None

            self.train_datasets = train_datasets
            self.test_datasets = test_datasets

        if not(self.eval):
            print('Model in training mode')
            self.log_file = os.path.join(self.ckpt_folder, "training_log.csv")
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f) 
                writer.writerow(["epoch", "training_loss", "validation_loss", "time"])
            self.train()
        else:
            print('Model in evaluation mode')
            self.set_eval_mode()

    def set_eval_mode(self):
        for modality in self.modalities:
            if modality == 'text':
                self.Encoders[modality].eval()
                self.Decoders[modality].eval()
            elif modality in ['vote', 'discrete_choice']:
                for t in self.Encoders[modality]:
                    self.Encoders[modality][t].eval()
                    if modality == 'vote':
                        self.Decoders[modality][t].eval()
                    else:
                        for k in self.Decoders[modality][t]:
                            self.Decoders[modality][t][k].eval()
        if self.labels_size != 0:
            self.predictor.eval()

    def train(self):
        """
        Train the model over multiple epochs, updating the model parameters based on the training data.
        
        This function handles the training loop, including loading the training and test datasets, 
        performing forward and backward passes, and logging the training and validation losses. 
        It also manages the saving of the best model based on validation loss and updates the 
        prior parameters if required. The training process is controlled by the number of epochs 
        specified, and the function ensures that the model is trained efficiently using mini-batch 
        gradient descent.

        Attributes:
            self.epochs (int): The current epoch number.
            self.num_epochs (int): The total number of epochs to train the model.
            self.ckpt_folder (str): The directory where model checkpoints are saved.
            self.log_every_n_epochs (int): Frequency of logging training progress.
            self.update_prior (bool): Flag indicating whether to update the prior parameters.
        """

        counter = 0
        best_loss = np.Inf
        best_epoch = -1
        best_state = None
        self.save_model("{}/best_model_{}.ckpt".format(self.ckpt_folder, self.model_name))
        
        for epoch in range(self.epochs, self.num_epochs):

            start_time = time.time()

            train_data_loaders = []
            for train_data in self.train_datasets:
                train_data_loader = DataLoader(
                    train_data,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )
                train_data_loaders.append(train_data_loader)

            if self.test_datasets is not None:
                test_data_loaders = []
                for test_data in self.test_datasets:
                    test_data_loader = DataLoader(
                        test_data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                    )
                    test_data_loaders.append(test_data_loader)

            training_loss = self.epoch(train_data_loaders, validation=False)

            if self.test_datasets is not None:
                validation_loss = self.epoch(test_data_loaders, validation=True)
            else:
                validation_loss = -1

            if (epoch + 1) % self.log_every_n_epochs == 0:
                save_name = f'{self.ckpt_folder}/IdealPointNN_K{self.n_dims}_{self.predictor_type}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{self.epochs+1}.ckpt'
                # a la fin d'une epoch on enregistrait systématique le modèle
                #self.save_model(save_name)

            if self.update_prior:
                list_ideology_covariates = [train_data.M_ideology_covariates for train_data in self.train_datasets]
                M_ideology_covariates = np.concatenate(list_ideology_covariates, axis=0)
                ideal_points = self.get_ideal_points(self.train_datasets)
                self.prior.update_parameters(
                    ideal_points, M_ideology_covariates
                )

            # Stopping rule for the optimization routine
            if self.test_datasets is not None:
                if validation_loss + self.delta < best_loss:
                    best_loss = validation_loss
                    best_epoch = self.epochs
                    best_state = self.get_model_state()
                    #self.save_model("{}/best_model_{}.ckpt".format(self.ckpt_folder, self.model_name))
                    counter = 0
                else:
                    counter += 1
            else:
                if training_loss + self.delta < best_loss:
                    best_loss = training_loss
                    best_epoch = self.epochs
                    best_state = self.get_model_state()
                    #self.save_model("{}/best_model_{}.ckpt".format(self.ckpt_folder, self.model_name))
                    counter = 0
                else:
                    counter += 1

            if counter >= self.patience:
                print(
                    "\nEarly stopping at Epoch {}. Reverting to Epoch {}".format(
                        self.epochs + 1, best_epoch + 1
                    )
                )
                #ckpt = "{}/best_model_{}.ckpt".format(self.ckpt_folder, self.model_name)
                self.load_model_state(best_state)
                break

            self.epochs += 1

            # Log the training and validation loss to CSV
            end_time = time.time()
            epoch_duration = end_time - start_time
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.epochs, training_loss, validation_loss, epoch_duration])


        # tout a la fin de l'entrainement, on charge les meilleurs paramètres du modèles (depuis best_state
        # et on sauvegarde le modèle une seule fois
        if best_state is not None:
            self.load_model_state(best_state)
            self.save_model("{}/best_model_{}.ckpt".format(self.ckpt_folder, self.model_name))


    def get_model_state(self):
        # recupère les paramètres du modèle
        state = {
            "Encoders": {},
            "Decoders": {},
            "predictor": None,
            "optimizer": self.optimizer.state_dict(),
            "epochs": self.epochs,
        }
        for modality in self.modalities:
            if modality == 'text':
                state["Encoders"][modality] = self.Encoders[modality].state_dict()
                state["Decoders"][modality] = self.Decoders[modality].state_dict()
            elif modality in ['vote', 'discrete_choice']:
                state["Encoders"][modality] = {}
                state["Decoders"][modality] = {}
                for t in self.Encoders[modality].keys():
                    state["Encoders"][modality][t] = self.Encoders[modality][t].state_dict()
                    state["Decoders"][modality][t] = self.Decoders[modality][t].state_dict()    
        
        if self.labels_size != 0:
            state["predictor"] = self.predictor.state_dict()

        return state    
                    

    def load_model_state(self, state):
        # charge les paramètres du modèle à partir d'un état donné
        for modality in self.modalities:
            if modality == 'text':
                self.Encoders[modality].load_state_dict(state["Encoders"][modality])
                self.Decoders[modality].load_state_dict(state["Decoders"][modality])
            elif modality in ['vote', 'discrete_choice']:
                for t in state["Encoders"][modality].keys():
                    self.Encoders[modality][t].load_state_dict(state["Encoders"][modality][t])
                    self.Decoders[modality][t].load_state_dict(state["Decoders"][modality][t])
        
        if self.labels_size != 0 and state["predictor"] is not None:
            self.predictor.load_state_dict(state["predictor"])

        self.optimizer.load_state_dict(state["optimizer"])
        self.epochs = state["epochs"]
        

    def epoch(self, data_loaders, validation=False):
        """
        Train the model for one epoch.
        """
        if validation:
            for modality in self.modalities:
                if modality == 'text':
                    self.Encoders[modality].eval()
                    self.Decoders[modality].eval()
                elif modality == 'vote':
                    for t in self.Encoders[modality].keys():
                        self.Encoders[modality][t].eval()
                        self.Decoders[modality][t].eval()
                elif modality == 'discrete_choice':
                    for t in self.Encoders[modality].keys():
                        for k2,v2 in self.Encoders[modality][t].items():
                            if k2 not in ["M_features", "M_colnames"]:
                                self.Encoders[modality][t].eval()
                                self.Decoders[modality][t][k2].eval()
            if self.labels_size != 0:
                self.predictor.eval()
        else:
            for modality in self.modalities:
                if modality == 'text':
                    self.Encoders[modality].train()
                    self.Decoders[modality].train()
                elif modality == 'vote':
                    for t in self.Encoders[modality].keys():
                        self.Encoders[modality][t].train()
                        self.Decoders[modality][t].train()
                elif modality == 'discrete_choice':
                    for t in self.Encoders[modality].keys():
                        self.Encoders[modality][t].train()
                        for k2,v2 in self.Decoders[modality][t].items():
                            if k2 not in ["M_features", "M_colnames"]:
                                self.Decoders[modality][t][k2].train()

            if self.labels_size != 0:
                self.predictor.train()

        epochloss_lst = []
        for j,data_loader in enumerate(data_loaders):
            for iter, data in enumerate(data_loader):

                t = int(data['t'][0])


                # Choose at random one modality to encode in and one modality to decode in (fast)
                enc_mod = random.choice(self.modalities)
                dec_mod = random.choice(self.modalities) 

                if not validation:
                    self.optimizer.zero_grad()

                # Unpack data
                for modality in [enc_mod, dec_mod]:
                    d = data[modality]
                    if modality in ['text', 'vote']:
                        for key, value in d.items():
                            d[key] = value.to(self.device)
                    elif modality == 'discrete_choice':
                        for key, value in d.items():
                            if key not in ["M_features"]:
                                d[key]['M_features'] = value['M_features'].to(self.device)
                            if key == "M_features":
                                d[key] = value.to(self.device)
                
                for k,v in data.items():
                    if k not in self.modalities:
                        if isinstance(v, torch.Tensor):
                            data[k] = v.clone().detach().to(self.device)
                        else:
                            data[k] = torch.tensor(v).to(self.device)
                        

                bows = data[enc_mod].get("M_features", None)
                bows = bows.reshape(bows.shape[0], -1)
                embeddings = data[enc_mod].get("M_embeddings", None)

                ideology_covariates = data.get("M_ideology_covariates", None)
                content_covariates = data[dec_mod].get("M_content_covariates", None)
                prediction_covariates = data.get("M_prediction", None)
                target_labels = data.get("M_labels", None)

                if embeddings is not None:
                    x_input = embeddings
                else:
                    x_input = bows

                if self.verbose:
                    print('------------------------')
                    print('enc_mod: ', enc_mod)
                    print('dec_mod: ', dec_mod)
                    print('t: ', t)
                    print('x_input0: ', x_input.shape)
                # Get theta and compute reconstruction loss
                if ideology_covariates is not None and self.encoder_include_ideology_covariates:
                    x_input = torch.cat((x_input, ideology_covariates), 1)

                if self.verbose: print('x_input1: ', x_input.shape)

                x_input = x_input.float()
                
                if enc_mod == 'text':
                    z = self.Encoders[enc_mod](x_input)
                elif enc_mod == 'vote':
                    z = self.Encoders[enc_mod][t](x_input)
                elif enc_mod == 'discrete_choice':
                    z = self.Encoders[enc_mod][t](x_input)

                if content_covariates is not None:
                    theta = torch.cat((z, content_covariates), 1)
                else:
                    theta = z

                reconstruction_loss = 0

                if 'vote' in self.modalities: # Impute missing values in the vote data
                    x_output = data['vote']["M_features"]
                    x_output = x_output.reshape(x_output.shape[0], -1).float().to(self.device)
                    x_recon_2 = self.Decoders['vote'][t](z)
                    x_recon_binary = (torch.sigmoid(x_recon_2) > 0.5).float().to(self.device)
                    missing_values = data['vote']['missing_values'].to(self.device)
                    x_output_imputed = torch.where(missing_values, x_recon_binary, x_output)
                    if validation:
                        self.test_datasets[j].data['vote']['M_features'][data['i'].cpu()] = x_output_imputed.cpu()
                    else:
                        self.train_datasets[j].data['vote']['M_features'][data['i'].cpu()] = x_output_imputed.cpu()

                if dec_mod == 'text':
                    x_output = data[dec_mod]["M_features"]
                    x_output = x_output.reshape(x_output.shape[0], -1)
                    x_recon = self.Decoders[dec_mod](theta)
                    #reconstruction_loss = F.cross_entropy(x_recon, x_output)
                    x_recon = torch.sigmoid(x_recon) #added pierre
                    
                    try:
                        #reconstruction_loss = nn.BCELoss()(x_recon, x_output) #added pierre
                        reconstruction_loss = nn.BCEWithLogitsLoss()(x_recon, x_output) #added pierre
                    except Exception as e:
                        print('Error in BCELoss: ', e)
                        print('max input: ', x_output.max())
                        print('min input: ', x_output.min())
                        print('max recon: ', x_recon.max().cpu().detach().numpy())
                        print('min recon: ', x_recon.min().cpu().detach().numpy())
                        # if x_recon contains NaNs, prints the number of NaNs
                        print('Number of NaNs in x_recon: ', torch.sum(torch.isnan(x_recon)).cpu().detach().numpy())
                        

                elif dec_mod == 'vote':
                    x_recon = x_recon_2
                    criterion = nn.BCEWithLogitsLoss()
                    reconstruction_loss = criterion(x_recon, x_output)
                elif dec_mod == 'discrete_choice':
                    columns = [k for k in list(data[dec_mod].keys()) if k != 'M_features']
                    for col in columns:
                        x_output = data[dec_mod][col]["M_features"]
                        x_output = x_output.reshape(x_output.shape[0], -1)
                        x_recon = self.Decoders[dec_mod][t][col](theta)
                        reconstruction_loss += F.cross_entropy(x_recon, x_output)

                # Get prior on theta and compute regularization loss
                theta_prior = self.prior.sample(
                    N=x_input.shape[0],
                    M_prevalence_covariates=ideology_covariates
                ).to(self.device)
                mmd_loss = MMD(z, theta_prior, device = self.device, kernel = 'multiscale')

                # Predict labels and compute prediction loss
                if target_labels is not None:
                    predictions = self.predictor(theta, prediction_covariates)
                    if self.predictor_type == "classifier":
                        target_labels = target_labels.squeeze()#.to(torch.int64)
                        prediction_loss = F.cross_entropy(predictions, target_labels)
                    elif self.predictor_type == "regressor":
                        prediction_loss = F.mse_loss(predictions, target_labels)
                else:
                    prediction_loss = 0
                
                # Total loss
                loss = (
                    reconstruction_loss
                    + mmd_loss * self.w_prior
                    + prediction_loss * self.w_pred_loss
                )

                if not validation:
                    loss.backward()
                    self.optimizer.step()

                epochloss_lst.append(loss.item())

                if (iter + 1) % self.print_every_n_batches == 0:
                    if validation:
                        print(
                            f"Epoch {(self.epochs+1):>3d}\tIter {(iter+1):>4d}\tMean Validation Loss:{loss.item():<.7f}\nMean Rec Loss:{reconstruction_loss.item():<.7f}\nMMD Loss:{mmd_loss.item()*self.w_prior:<.7f}\nMean Pred Loss:{prediction_loss*self.w_pred_loss:<.7f}\n"
                        )
                    else:
                        print(
                            f"Epoch {(self.epochs+1):>3d}\tIter {(iter+1):>4d}\tMean Training Loss:{loss.item():<.7f}\nMean Rec Loss:{reconstruction_loss.item():<.7f}\nMMD Loss:{mmd_loss.item()*self.w_prior:<.7f}\nMean Pred Loss:{prediction_loss*self.w_pred_loss:<.7f}\n"
                        )

        if (self.epochs + 1) % self.print_every_n_epochs == 0:
            if validation:
                print(
                    f"\nEpoch {(self.epochs+1):>3d}\tMean Validation Loss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}\n"
                )
            else:
                print(
                    f"\nEpoch {(self.epochs+1):>3d}\tMean Training Loss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}\n"
                )

            if self.print_dims:
                pass

            if self.content_covariate_size != 0 and self.print_content_covariates:
                pass

        return sum(epochloss_lst)/len(epochloss_lst)
    

    def get_ideal_points(self, datasets, modality=None):
        """
        Compute the ideal points for the documents in the dataset.
        """

        ideal_points = []
        for dataset in datasets:
            
            t = dataset.t

            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            if modality is None: 
                modality = self.ref_modality

            for data in data_loader:
                d = data[modality]

                if modality in ['text', 'vote']:
                    for key, value in d.items():
                        d[key] = value.to(self.device)
                elif modality == 'discrete_choice':
                    for key, value in d.items():
                        if key not in ["M_features"]:
                            d[key]['M_features'] = value['M_features'].to(self.device)
                        if key == "M_features":
                            d[key] = value.to(self.device)
                
                for k,v in data.items():
                    if k not in self.modalities:
                        data[k] = v.to(self.device)

                bows = data[modality].get("M_features", None)
                bows = bows.reshape(bows.shape[0], -1)
                embeddings = data[modality].get("M_embeddings", None)
                ideology_covariates = data.get("M_ideology_covariates", None)

                if embeddings is not None:
                    x_input = embeddings
                else:
                    x_input = bows

                if ideology_covariates is not None and self.encoder_include_ideology_covariates:
                    x_input = torch.cat((x_input, ideology_covariates), 1)

                x_input = x_input.float()

                if modality == 'text':
                    z = self.Encoders[modality](x_input)   
                elif modality == 'vote':
                    z = self.Encoders[modality][t](x_input)
                elif modality == 'discrete_choice':
                    z = self.Encoders[modality][t](x_input)

                ideal_points.append(z.detach().cpu().numpy())

        ideal_points = np.concatenate(ideal_points, axis=0)

        return ideal_points

    def save_model(self, save_name):
        if self.ignore_saves_loads:
            return None
        
        encoders = {}
        decoders = {}
        for modality in self.modalities:
            if modality == 'text':
                encoders[modality] = self.Encoders[modality].state_dict()
                decoders[modality] = self.Decoders[modality].state_dict()
            elif modality in ['vote', 'discrete_choice']:
                encoders[modality] = {}
                decoders[modality] = {}
                for t in self.Encoders[modality].keys():
                    encoders[modality][t] = self.Encoders[modality][t].state_dict()
                    decoders[modality][t] = self.Decoders[modality][t].state_dict()

        if self.labels_size != 0:
            predictor_state_dict = self.predictor.state_dict()
        else:
            predictor_state_dict = None
        optimizer_state_dict = self.optimizer.state_dict()

        all_vars = vars(self)

        checkpoint = {}
        for key, value in all_vars.items():
            if key not in ["Encoders", "Decoders", "predictor", "optimizer"]:
                checkpoint[key] = value

        checkpoint["Encoders"] = encoders
        checkpoint["Decoders"] = decoders
        if self.labels_size != 0:
            checkpoint["predictor"] = predictor_state_dict
        checkpoint["optimizer"] = optimizer_state_dict

        print(f"Saving model to {save_name}")
        torch.save(checkpoint, save_name)

    def load_model(self, ckpt, map_location=None):
        #
        """
        Helper function to load a GTM model.
        """
        ckpt = torch.load(ckpt, map_location=map_location)
        current_eval_status = self.eval # ici on garde l'ancien status de eval
        for key, value in ckpt.items():
            if key not in ["Encoders", "Decoders", "predictor", "optimizer", "eval", 'input_size', 'bow_size',
                           'verbose', 'device']:
                setattr(self, key, value)
                if self.verbose:
                    print(f"Loading {key} from checkpoint")
        self.eval = current_eval_status #on lui remet la variable eval pour empecher l'ecrasement

        # Load Encoders
        if not hasattr(self, "Encoders"):
            self.Encoders = {}
        for modality in self.modalities:
            if modality == 'text':
                if modality not in self.Encoders:
                    self.Encoders[modality] = self._create_encoder(modality)
                self.Encoders[modality].load_state_dict(ckpt["Encoders"][modality])
            elif modality in ['vote', 'discrete_choice']:
                self.Encoders[modality] = {}
                for t, state_dict in ckpt["Encoders"][modality].items():
                    if t not in self.Encoders[modality]:
                        
                        t_idx = list(ckpt["Encoders"][modality].keys()).index(t)
                        # On garde une trace de l'index temporel (t_idx)
                        assert self.train_datasets[t_idx].t == t, f"t: {t} != self.train_datasets[t_idx].t: {self.train_datasets[t_idx].t}"
                        
                        features_size = self.train_datasets[t_idx].data[modality]['M_features'].shape[1]
                        bow_size = features_size
                        input_size = features_size

                        if self.verbose:
                            print(f'{t} not in self.Encoders[{modality}], creating it')
                        self.Encoders[modality][t] = self._create_encoder(modality, input_size)
                    self.Encoders[modality][t].load_state_dict(state_dict)

        # Load Decoders
        if not hasattr(self, "Decoders"):
            self.Decoders = {}
        for modality in self.modalities:
            if modality == 'text':
                if modality not in self.Decoders:
                    self.Decoders[modality] = self._create_decoder(modality)
                self.Decoders[modality].load_state_dict(ckpt["Decoders"][modality])
            elif modality == 'vote':
                self.Decoders[modality] = {}
                for t, state_dict in ckpt["Decoders"][modality].items():
                    if t not in self.Decoders[modality]:
                        
                        t_idx = list(ckpt["Decoders"][modality].keys()).index(t)
                        assert self.train_datasets[t_idx].t == t, f"t: {t} != self.train_datasets[t_idx].t: {self.train_datasets[t_idx].t}"

                        features_size = self.train_datasets[t_idx].data[modality]['M_features'].shape[1]
                        bow_size = features_size
                        input_size = features_size

                        if self.verbose:
                            print(f'{t} not in self.Decoders[{modality}], creating it')
                        self.Decoders[modality][t] = self._create_decoder(modality, bow_size)
                    self.Decoders[modality][t].load_state_dict(state_dict)
            elif modality == 'discrete_choice':
                self.Decoders[modality] = {}
                for t, var_dict in ckpt["Decoders"][modality].items():
                    self.Decoders[modality][t] = {}
                    for k, state_dict in var_dict.items():
                        if k not in self.Decoders[modality][t]:
                            self.Decoders[modality][t][k] = self._create_decoder(modality)
                        self.Decoders[modality][t][k].load_state_dict(state_dict)

        # Load predictor
        if self.labels_size != 0:
            if not hasattr(self, "predictor"):
                self.predictor = self._create_predictor()
            self.predictor.load_state_dict(ckpt["predictor"])

        # Load optimizer
        if not hasattr(self, "optimizer"):
            self.optimizer = self._create_optimizer()
            self.optimizer.load_state_dict(ckpt["optimizer"])

    def _create_encoder(self, 
                        modality,
                        input_size,):
        
        encoder_dims = [input_size + self.ideology_covariate_size] if self.encoder_include_ideology_covariates else [input_size]

        encoder_dims.extend(self.encoder_params[modality]['encoder_hidden_layers'])
        encoder_dims.extend([self.n_dims])
        if self.verbose:
            print('New, created encoder_dims: ', encoder_dims)
        return EncoderMLP(
            encoder_dims=encoder_dims,
            encoder_non_linear_activation=self.encoder_params[modality]['encoder_non_linear_activation'],
            encoder_bias=self.encoder_params[modality]['encoder_bias'],
            dropout=self.dropout,
        ).to(self.device)

    def _create_decoder(self, modality,
                        bow_size):
        
        decoder_dims = [self.n_dims + self.content_covariate_size[modality]]
        decoder_dims.extend(self.decoder_params[modality]['decoder_hidden_layers'])
        decoder_dims.extend([bow_size])
        return DecoderMLP(
            decoder_dims=decoder_dims,
            decoder_non_linear_activation=self.decoder_params[modality]['decoder_non_linear_activation'],
            decoder_bias=self.decoder_params[modality]['decoder_bias'],
            dropout=self.dropout,
        ).to(self.device)

    def _create_predictor(self):
        predictor_dims = [self.n_dims + self.train_datasets[0].M_prediction_covariates.shape[1]]
        predictor_dims.extend(self.predictor_hidden_layers)
        predictor_dims.extend([self.labels_size])
        return Predictor(
            predictor_dims=predictor_dims,
            predictor_non_linear_activation=self.predictor_non_linear_activation,
            predictor_bias=self.predictor_bias,
            dropout=self.dropout,
        ).to(self.device)

    def _create_optimizer(self):
        params = []
        for modality in self.modalities:
            if modality == 'text':
                params.extend(list(self.Encoders[modality].parameters()))
                params.extend(list(self.Decoders[modality].parameters()))
            elif modality in ['vote', 'discrete_choice']:
                for t in self.Encoders[modality]:
                    params.extend(list(self.Encoders[modality][t].parameters()))
                for t in self.Decoders[modality]:
                    if modality == 'vote':
                        params.extend(list(self.Decoders[modality][t].parameters()))
                    else:
                        for k in self.Decoders[modality][t]:
                            params.extend(list(self.Decoders[modality][t][k].parameters()))
        if self.labels_size != 0:
            params.extend(list(self.predictor.parameters()))
        return torch.optim.Adam(params, lr=self.learning_rate)

    def to(self, device):
        """
        Move the model to a different device.
        """
        for modality in self.modalities:
            self.Encoders[modality].to(device)
            self.Decoders[modality].to(device)
        self.prior.to(device)
        if self.labels_size != 0:
            self.predictor.to(device)
        self.device = device
