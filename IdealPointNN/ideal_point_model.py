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

encoder_args = {
    "encoder_input": "bow",
    "encoder_hidden_layers":[256],
    "encoder_non_linear_activation":"relu",
    "encoder_bias":True
}

decoder_args = {
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

predictor_args = {
    "predictor_type": "classifier",
    "predictor_hidden_layers":[],
    "predictor_non_linear_activation":None,
    "predictor_bias":True
}


class IdealPointNN:
    """
    Wrapper class for the Neural Ideal Point Model with a single encoder.
    """

    def __init__(
        self,
        train_datasets=None,
        test_datasets=None,
        n_dims=1,
        update_prior=False,
        tol=0.001,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
        predictor_args=predictor_args,
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
        w_vote=1,
        w_pred_loss=1,
        ckpt_folder="../ckpt",
        ckpt=None,
        device=None,
        seed=42,
    ):
        """
        Args:
            (Args remain unchanged...)
        """

        if ckpt:
            self.load_model(ckpt)

        else:
            train_data = train_datasets[0]

            self.n_dims = n_dims
            self.update_prior = update_prior
            self.tol = tol

            # Unified encoder parameters
            self.encoder_args = encoder_args

            self.encoder_hidden_layers = encoder_args['encoder_hidden_layers']
            self.encoder_non_linear_activation = encoder_args['encoder_non_linear_activation']
            self.encoder_bias = encoder_args['encoder_bias']

            self.decoder_args = decoder_args
            self.predictor_args = predictor_args

            self.device = device
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
            self.w_vote = w_vote
            self.w_pred_loss = w_pred_loss
            self.ckpt_folder = ckpt_folder

            self.predictor_type = self.predictor_args['predictor_type']
            self.predictor_hidden_layers = self.predictor_args['predictor_hidden_layers']
            self.predictor_non_linear_activation = self.predictor_args['predictor_non_linear_activation']
            self.predictor_bias = self.predictor_args['predictor_bias']

            if train_data.labels is not None:
                labels_size = train_data.M_labels.shape[1]
                if self.predictor_type == "classifier":
                    labels_size = len(np.unique(train_data.M_labels))
                else:
                    labels_size = 1
            else:
                labels_size = 0

            self.labels_size = labels_size

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

            # Concatenate all modalities for unified input
            self.modalities = train_data.modalities
            self.ref_modality = self.modalities[0]

            total_input_size = 0
            for modality in self.modalities:
                if modality =='text':
                    encoder_input = self.encoder_args['encoder_input']
                    if encoder_input == "embeddings":
                        features_size = train_data.data[modality]['M_embeddings'].shape[1]
                    else:
                        features_size = train_data.data[modality]['M_features'].shape[1]
                else:
                    features_size = train_data.data[modality]['M_features'].shape[1]                 

                total_input_size += features_size
                self.input_size = total_input_size

            if train_data.ideology is not None:
                ideology_covariate_size = train_data.M_ideology_covariates.shape[1]
            else:
                ideology_covariate_size = 0

            total_input_size += ideology_covariate_size

            # Single Encoder for concatenated modalities
            encoder_dims = [total_input_size]
            encoder_dims.extend(self.encoder_hidden_layers)
            encoder_dims.extend([n_dims])

            self.Encoder = EncoderMLP(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=self.encoder_non_linear_activation,
                encoder_bias=self.encoder_bias,
                dropout=self.dropout,
            ).to(self.device)

            self.Decoders = {}
            for mod in self.modalities:
                decoder_hidden_layers = self.decoder_args[mod]['decoder_hidden_layers']
                decoder_non_linear_activation = self.decoder_args[mod]['decoder_non_linear_activation']
                decoder_bias = self.decoder_args[mod]['decoder_bias']

                if mod == 'text':
                    if self.encoder_args['encoder_input'] == "embeddings":
                        features_size = train_data.data[mod]['M_embeddings'].shape[1]
                    else:
                        features_size = train_data.data[mod]['M_features'].shape[1] 
                else:
                    features_size = train_data.data[mod]['M_features'].shape[1]
                if train_data.data[mod].get('M_content_covariates', None) is not None:
                    content_covariate_size = train_data.data[mod]['M_content_covariates'].shape[1]
                else:
                    content_covariate_size = 0

                decoder_dims = [n_dims + content_covariate_size]
                decoder_dims.extend(decoder_hidden_layers)
                decoder_dims.extend([features_size])

                self.Decoders[mod] = DecoderMLP(
                    decoder_dims=decoder_dims,
                    decoder_non_linear_activation=decoder_non_linear_activation,
                    decoder_bias=decoder_bias,
                    dropout=dropout,
                ).to(self.device)

            # Define Prior
            self.prior = NormalPrior(
                ideology_covariate_size,
                n_dims,
                device=self.device,
            )

            # Define Predictor if there are labels
            if train_data.labels is not None:
                labels_size = train_data.M_labels.shape[1]
                predictor_dims = [n_dims]
                predictor_dims.extend(self.predictor_hidden_layers)
                predictor_dims.extend([labels_size])

                self.predictor = Predictor(
                    predictor_dims=predictor_dims,
                    predictor_non_linear_activation=self.predictor_args['predictor_non_linear_activation'],
                    predictor_bias=self.predictor_args['predictor_bias'],
                    dropout=self.dropout,
                ).to(self.device)

            # Optimizer for the unified encoder and decoders
            self.optimizer = torch.optim.Adam(
                list(self.Encoder.parameters()) +
                [param for decoder in self.Decoders.values() for param in decoder.parameters()] +
                (list(self.predictor.parameters()) if train_data.labels is not None else []),
                lr=self.learning_rate,
            )

            self.epochs = 0
            self.reconstruction_loss = None
            self.mmd_loss = None
            self.prediction_loss = None

            self.train_datasets = train_datasets
            self.test_datasets = test_datasets

            self.train()

    def train(self):
        """
        Train the model.
        """

        counter = 0
        best_loss = np.Inf
        best_epoch = -1
        self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
        
        for epoch in range(self.epochs, self.num_epochs):

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

            if (epoch + 1) % self.log_every_n_epochs == 0:
                save_name = f'{self.ckpt_folder}/IdealPointNN_K{self.n_dims}_{self.predictor_type}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{self.epochs+1}.ckpt'
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
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    counter = 0
                else:
                    counter += 1
            else:
                if training_loss + self.delta < best_loss:
                    best_loss = training_loss
                    best_epoch = self.epochs
                    self.save_model("{}/best_model.ckpt".format(self.ckpt_folder))
                    counter = 0
                else:
                    counter += 1

            if counter >= self.patience:
                print(
                    "\nEarly stopping at Epoch {}. Reverting to Epoch {}".format(
                        self.epochs + 1, best_epoch + 1
                    )
                )
                ckpt = "{}/best_model.ckpt".format(self.ckpt_folder)
                self.load_model(ckpt)
                break

            self.epochs += 1

    def epoch(self, data_loaders, validation=False):
        """
        Train the model for one epoch, encoding all modalities together and decoding each separately.
        """
        if validation:
            self.Encoder.eval()
            for modality in self.modalities:
                self.Decoders[modality].eval()
            if self.labels_size != 0:
                self.predictor.eval()
        else:
            self.Encoder.train()
            for modality in self.modalities:
                self.Decoders[modality].train()
            if self.labels_size != 0:
                self.predictor.train()

        epoch_loss_lst = []

        for j, data_loader in enumerate(data_loaders):
            for iter, data in enumerate(data_loader):
                if not validation:
                    self.optimizer.zero_grad()

                # Unpack data and send to device
                input_features = []
                for modality in self.modalities:
                    modality_data = data[modality]
                    M_embeddings = modality_data.get("M_embeddings", None)
                    if M_embeddings is not None:
                        modality_features = M_embeddings.to(self.device).float()
                    else:
                        modality_features = modality_data.get("M_features", None).to(self.device).float()
                    modality_features = modality_features.reshape(modality_features.shape[0], -1)
                    input_features.append(modality_features)

                # Concatenate features from all modalities
                x_input = torch.cat(input_features, dim=1)

                ideology_covariates = data.get("M_ideology_covariates", None)
                if ideology_covariates is not None:
                    ideology_covariates = ideology_covariates.to(self.device).float()
                    x_input = torch.cat((x_input, ideology_covariates), 1)

                x_input = x_input.float()

                # Encode the concatenated input using the unified encoder
                z = self.Encoder(x_input)

                total_reconstruction_loss = 0

                # Decode each modality separately
                for modality in self.modalities:
                    modality_data = data[modality]
                    content_covariates = modality_data.get("M_content_covariates", None)
                    if content_covariates is not None:
                        content_covariates = content_covariates.to(self.device).float()
                        theta = torch.cat((z, content_covariates), 1)
                    else:
                        theta = z

                    # Compute reconstruction loss for the modality
                    if modality == 'text':
                        x_output = modality_data.get("M_embeddings", None)
                        if x_output is None:
                            x_output = modality_data["M_features"].to(self.device).float()
                            x_output = x_output.reshape(x_output.shape[0], -1)
                            x_recon = self.Decoders[modality](theta)
                            reconstruction_loss = F.cross_entropy(x_recon, x_output)
                        else:
                            x_output = x_output.to(self.device).float()
                            x_output = x_output.reshape(x_output.shape[0], -1)
                            x_recon = self.Decoders[modality](theta)
                            reconstruction_loss = F.mse_loss(x_recon, x_output)
                    elif modality == 'vote':
                        x_output = modality_data["M_features"].to(self.device).float()
                        x_recon = self.Decoders[modality](theta)
                        criterion = nn.BCEWithLogitsLoss()
                        reconstruction_loss = criterion(x_recon, x_output)*self.w_vote
                    elif modality == 'discrete_choice':
                        for k2 in self.Decoders[modality].keys():
                            x_output = modality_data[k2]["M_features"].to(self.device).float()
                            x_output = x_output.reshape(x_output.shape[0], -1)
                            x_recon = self.Decoders[modality][k2](theta)
                            reconstruction_loss += F.cross_entropy(x_recon, x_output)

                    total_reconstruction_loss += reconstruction_loss

                # Compute MMD loss (regularization)
                theta_prior = self.prior.sample(
                    N=x_input.shape[0],
                    M_prevalence_covariates=ideology_covariates
                ).to(self.device)
                mmd_loss = MMD(z, theta_prior, device=self.device, kernel='multiscale')

                # Compute prediction loss if there are labels
                target_labels = data.get("M_labels", None)
                prediction_loss = 0
                if target_labels is not None:
                    target_labels = target_labels.to(self.device)
                    prediction_covariates = data.get("M_prediction_covariates", None)
                    if prediction_covariates is not None:
                        prediction_covariates = prediction_covariates.to(self.device)

                    predictions = self.predictor(z, prediction_covariates)

                    if self.predictor_type == "classifier":
                        #target_labels = target_labels.view(-1).to(torch.int64)
                        
                        # Debugging prints to ensure correctness
                        #print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
                        #print(f"Target labels shape: {target_labels.shape}, dtype: {target_labels.dtype}")
                        #print(f"Predictions range: [{predictions.min().item()}, {predictions.max().item()}]")
                        #print(f"Target labels range: [{target_labels.min().item()}, {target_labels.max().item()}]")

                        prediction_loss = F.cross_entropy(predictions, target_labels)

                    elif self.predictor_type == "regressor":
                        prediction_loss = F.mse_loss(predictions, target_labels)

                # Total loss calculation
                loss = (
                    total_reconstruction_loss
                    + mmd_loss * self.w_prior
                    + prediction_loss * self.w_pred_loss
                )

                # Backpropagation and optimization step if not in validation
                if not validation:
                    loss.backward()
                    self.optimizer.step()

                epoch_loss_lst.append(loss.item())

                # Print training/validation stats every n batches
                if (iter + 1) % self.print_every_n_batches == 0:
                    print(
                        f"Epoch {(self.epochs + 1):>3d}\tIter {(iter + 1):>4d}\t"
                        f"Loss: {loss.item():<.7f}\t"
                        f"Rec Loss: {total_reconstruction_loss.item():<.7f}\t"
                        f"MMD Loss: {mmd_loss.item() * self.w_prior:<.7f}\t"
                        f"Pred Loss: {prediction_loss * self.w_pred_loss:<.7f}\n"
                    )

        # Print stats after each epoch
        print(f"Epoch {(self.epochs + 1):>3d}\tMean Loss: {sum(epoch_loss_lst) / len(epoch_loss_lst):<.7f}\n")

        return sum(epoch_loss_lst)
    

    def get_ideal_points(self, datasets):
        """
        Compute the ideal points for the documents in the dataset by encoding all modalities together.
        """
        
        # Set the encoder and decoders to evaluation mode
        self.Encoder.eval()
        for modality in self.modalities:
            self.Decoders[modality].eval()

        ideal_points = []

        with torch.no_grad():
            for dataset in datasets:
                
                data_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )

                for data in data_loader:
                    # Unpack data and send to device
                    input_features = []
                    for modality in self.modalities:
                        modality_data = data[modality]
                        M_embeddings = modality_data.get("M_embeddings", None)
                        if M_embeddings is not None:
                            modality_features = M_embeddings.to(self.device)
                        else:
                            modality_features = modality_data.get("M_features", None).to(self.device)
                        modality_features = modality_features.reshape(modality_features.shape[0], -1)
                        input_features.append(modality_features)

                    # Concatenate features from all modalities
                    x_input = torch.cat(input_features, dim=1)

                    # Add ideology covariates if present
                    ideology_covariates = data.get("M_ideology_covariates", None)
                    if ideology_covariates is not None:
                        ideology_covariates = ideology_covariates.to(self.device)
                        x_input = torch.cat((x_input, ideology_covariates), 1)

                    x_input = x_input.float()

                    # Get the ideal points by passing the concatenated input through the unified encoder
                    z = self.Encoder(x_input)

                    ideal_points.append(z.detach().cpu().numpy())

            # Concatenate all ideal points collected across batches
            ideal_points = np.concatenate(ideal_points, axis=0)

        return ideal_points

    def save_model(self, save_name):
        """
        Save the model, including the single encoder, modality-specific decoders, predictor, and optimizer.
        """
        # Save the unified encoder's state
        encoder_state_dict = self.Encoder.state_dict()

        # Save each modality's decoder state
        decoders = {}
        for modality in self.modalities:
            decoders[modality] = self.Decoders[modality].state_dict()

        # Save the predictor's state (if applicable)
        if self.labels_size != 0:
            predictor_state_dict = self.predictor.state_dict()
        else:
            predictor_state_dict = None

        # Save the optimizer's state
        optimizer_state_dict = self.optimizer.state_dict()

        # Save all other model variables except the actual model components
        all_vars = vars(self)
        checkpoint = {}
        for key, value in all_vars.items():
            if key not in ["Encoder", "Decoders", "predictor", "optimizer"]:
                checkpoint[key] = value

        # Add the state dicts of the encoder, decoders, predictor, and optimizer to the checkpoint
        checkpoint["Encoder"] = encoder_state_dict
        checkpoint["Decoders"] = decoders
        if self.labels_size != 0:
            checkpoint["predictor"] = predictor_state_dict
        checkpoint["optimizer"] = optimizer_state_dict

        # Save the checkpoint to the specified file
        torch.save(checkpoint, save_name)


    def load_model(self, ckpt):
        """
        Load the model from a checkpoint file, including the unified encoder, modality-specific decoders, predictor, and optimizer.
        """
        ckpt = torch.load(ckpt)

        # Load all other model variables
        for key, value in ckpt.items():
            if key not in ["Encoder", "Decoders", "predictor", "optimizer"]:
                setattr(self, key, value)

        # Load the unified encoder's state dict
        if not hasattr(self, "Encoder"):
            total_input_size = sum([self.train_datasets[0].data[mod]['M_features'].shape[1] for mod in self.modalities])
            if self.encoder_include_ideology_covariates:
                total_input_size += self.ideology_covariate_size

            encoder_dims = [total_input_size] + self.encoder_hidden_layers + [self.n_dims]
            self.Encoder = EncoderMLP(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=self.encoder_non_linear_activation,
                encoder_bias=self.encoder_bias,
                dropout=self.dropout,
            ).to(self.device)

        self.Encoder.load_state_dict(ckpt["Encoder"])

        # Load each modality's decoder state dict
        if not hasattr(self, "Decoders"):
            for modality in self.modalities:
                decoder_dims = [self.n_dims + self.content_covariate_size[modality]]
                decoder_dims.extend(self.decoder_args[modality]["decoder_hidden_layers"])
                decoder_dims.extend([self.train_datasets[0].data[modality]["M_features"].shape[1]])

                self.Decoders[modality] = DecoderMLP(
                    decoder_dims=decoder_dims,
                    decoder_non_linear_activation=self.decoder_args[modality]["decoder_non_linear_activation"],
                    decoder_bias=self.decoder_args[modality]["decoder_bias"],
                    dropout=self.dropout,
                ).to(self.device)

        for modality in self.modalities:
            self.Decoders[modality].load_state_dict(ckpt["Decoders"][modality])

        # Load the predictor's state dict if applicable
        if self.labels_size != 0:
            if not hasattr(self, "predictor"):
                predictor_dims = [self.n_dims + self.prediction_covariate_size] + self.predictor_hidden_layers + [self.labels_size]
                self.predictor = Predictor(
                    predictor_dims=predictor_dims,
                    predictor_non_linear_activation=self.predictor_non_linear_activation,
                    predictor_bias=self.predictor_bias,
                    dropout=self.dropout,
                ).to(self.device)
            self.predictor.load_state_dict(ckpt["predictor"])

        # Load the optimizer's state dict
        if not hasattr(self, "optimizer"):
            list_of_encoder_parameters = list(self.Encoder.parameters())
            list_of_decoder_parameters = [list(self.Decoders[modality].parameters()) for modality in self.modalities]

            if self.labels_size != 0:
                list_of_predictor_parameters = list(self.predictor.parameters())
                self.optimizer = torch.optim.Adam(
                    list_of_encoder_parameters + list_of_decoder_parameters + list_of_predictor_parameters,
                    lr=self.learning_rate,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    list_of_encoder_parameters + list_of_decoder_parameters,
                    lr=self.learning_rate,
                )

        self.optimizer.load_state_dict(ckpt["optimizer"])

    def to(self, device):
        """
        Move the model components to a different device (CPU or GPU).
        """
        # Move the unified encoder to the specified device
        self.Encoder.to(device)

        # Move each modality's decoder to the specified device
        for modality in self.modalities:
            self.Decoders[modality].to(device)

        # Move the prior to the specified device
        self.prior.to(device)

        # Move the predictor to the specified device, if applicable
        if self.labels_size != 0:
            self.predictor.to(device)

        # Update the device attribute
        self.device = device
