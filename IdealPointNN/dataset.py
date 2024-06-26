#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from utils import bert_embeddings_from_list


class IdealPointNNDataset(Dataset):
    """
    Dataset for the IdealPointNN() model.
    """

    def __init__(
        self,
        df,
        time_col=None,
        time_value=None,
        ideology=None,
        prediction=None,
        labels=None,
        device=None,
    ):
        """
        Initialize GTMCorpus.

        Args:
            df : pandas DataFrame. For bag of words, must also contain 'doc_clean' with the cleaned text of each document. For embeddings, must contain a column 'doc' with the raw text of each document.
            ideology : string, formula for ideology covariates (of the form "~ cov1 + cov2 + ..."), but allows for transformations of e.g., "~ g(cov1) + h(cov2) + ...)". Use "C(your_categorical_variable)" to indicate a categorical variable. See the Patsy package for more details.
            prediction : string, formula for covariates used as inputs for the prediction task (also of the form "~ cov1 + cov2 + ..."). See the Patsy package for more details.
            labels : string, formula for labels used as outcomes for the prediction task (of the form "~ label1 + label2 + ...")
            device : string, device to use for SentenceTranformer embeddings
        """

        # Basic params and formulas
        self.modalities = None
        self.time_col = time_col
        self.time_value = time_value
        self.ideology = ideology
        self.prediction = prediction
        self.labels = labels
        self.device = device

        # Extract ideology covariates matrix
        if ideology is not None:
            self.ideology_colnames, self.M_ideology_covariates = self._transform_df(
                ideology, df
            )
        else:
            self.M_ideology_covariates = np.zeros(
                (len(df.index), 1), dtype=np.float32
            )

        # Extract prediction covariates matrix
        if prediction is not None:
            self.prediction_colnames, self.M_prediction = self._transform_df(
                prediction, df
            )
        else:
            self.M_prediction = None

        # Extract labels matrix
        if labels is not None:
            self.labels_colnames, self.M_labels = self._transform_df(
                labels, df
            )
        else:
            self.M_labels = None

        self.modalities = None

        if time_col is not None:
            self.df = df[df[time_col] == time_value].reset_index()
            if ideology is not None:
                self.M_ideology_covariates = self.M_ideology_covariates[df[time_col] == time_value]
            if prediction is not None:
                self.M_prediction = self.M_prediction[df[time_col] == time_value]
            if labels is not None:
                self.M_labels = self.M_labels[df[time_col] == time_value]
        else:
            self.df = df

        if time_value is None:
            self.t = 1
        else:
            self.t = time_value

        self.data = {}

    def add_modality(
        self,
        df,
        modality='text', 
        columns=None,
        content=None,
        vectorizer=None,
        vectorizer_args={},
        sbert_model_to_load=None,
        batch_size=64,
        max_seq_length=100000
        ):

        """
        Add a modality to the dataset.

        Args:
        modality : string, type of modality. Can be 'text' or 'discrete_choice'
        columns : list of strings, columns to use for the modality
        vectorizer : sklearn CountVectorizer object, if None, a new one will be created
        vectorizer_args: dict, arguments for the CountVectorizer object
        sbert_model_to_load : string, name of the SentenceTranformer model to load
        batch_size : int, batch size for SentenceTranformer embeddings
        max_seq_length : int, maximum sequence length for SentenceTranformer embeddings
        """

        d = {}

        if modality == 'text':
            
            # Compute bag of words matrix
            if vectorizer is None:
                d['vectorizer'] = CountVectorizer(**vectorizer_args)
                d['M_features'] = d['vectorizer'].fit_transform(df["doc_clean"])
            else:
                d['vectorizer'] = vectorizer
                d['M_features'] = d['vectorizer'].transform(df["doc_clean"])
            d['vocab'] = d['vectorizer'].get_feature_names_out()
            d['id2token'] = {
                k: v for k, v in zip(range(0, len(d['vocab'])), d['vocab'])
            }

            if sbert_model_to_load is not None:
                # Create embeddings matrix
                d['M_embeddings'] = None
                d['V_embeddings'] = None

                d['M_embeddings'] = bert_embeddings_from_list(
                    df["doc"], sbert_model_to_load, batch_size, max_seq_length, self.device
                )
                d['V_embeddings'] = bert_embeddings_from_list(
                    d['vocab'], sbert_model_to_load, batch_size, max_seq_length, self.device
                )

        elif modality == "discrete_choice":

            # M_features matrix for the encoder
            d['M_colnames'], d['M_features'] = self._transform_df(
                "~ {}".format(" + ".join(["C({})".format(col) for col in columns])), 
                df
                )

            # Separate M_features matrix for each variable for the decoders
            for column in columns:
                d[column] = {}
                d[column]['M_colnames'], d[column]['M_features'] = self._transform_df("~ C({}) - 1".format(column), df)

        elif modality == "vote":

            d['M_features'] = np.array(df[columns])    
            d['missing_values'] = np.isnan(d['M_features'])    
            d['M_features'] = np.nan_to_num(d['M_features'])

        # Extract content covariates matrix
        if content is not None:
            d['content_colnames'], d['M_content_covariates'] = self._transform_df(
                content, df
            )
            if self.time_col is not None:
                d['M_content_covariates'] = d['M_content_covariates'][df[self.time_col] == self.time_value]
            else:
                d['M_content_covariates'] = d['M_content_covariates']
        
        self.data[modality] = d

        self.modalities = list(self.data.keys())

    def _transform_df(self, formula, df):
        """
        Uses the patsy package to transform covariates into appropriate matrices
        """

        M = dmatrix(formula, df)
        colnames = M.design_info.column_names
        M = np.asarray(M, dtype=np.float32)

        return colnames, M

    def __len__(self):
        """Return length of dataset."""
        return len(self.df)

    def __getitem__(self, i):
        """Return sample from dataset at index i"""

        d = {}

        d['i'] = i
        d['t'] = self.t

        for mod in self.modalities:

            d[mod] = {}

            if mod == 'text':

                if type(self.data[mod]['M_features'][i]) == scipy.sparse.csr_matrix:
                    M_features_sample = torch.FloatTensor(self.data[mod]['M_features'][i].todense())
                else:
                    M_features_sample = torch.FloatTensor(self.data[mod]['M_features'][i])

                d[mod]["M_features"] = M_features_sample

            if self.data[mod].get('M_embeddings', None) is not None:
                d[mod]["M_embeddings"] = self.data[mod]['M_embeddings'][i]

            if mod == 'vote':
                d[mod]["M_features"] = self.data[mod]['M_features'][i]
                d[mod]['missing_values'] = self.data[mod]['missing_values'][i]

            if mod == 'discrete_choice':
                d[mod]["M_features"] = self.data[mod]['M_features'][i]
                for column in self.data[mod].keys():
                    if column not in ['M_colnames', 'M_features']:
                        d[mod][column] = {}
                        d[mod][column]["M_features"] = self.data[mod][column]['M_features'][i]

            if self.data[mod].get("M_content_covariates", None) is not None:
                d[mod]["M_content_covariates"] = self.data[mod]['M_content_covariates'][i]

        if self.ideology is not None:
            d["M_ideology_covariates"] = self.M_ideology_covariates[i]

        if self.prediction is not None:
            d["M_prediction"] = self.M_prediction[i]

        if self.labels is not None:
            d["M_labels"] = self.M_labels[i]

        return d