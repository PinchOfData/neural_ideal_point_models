#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import warnings
from typing import List, Optional
from tqdm import tqdm
import spacy
from spacy.cli import download as spacy_download
import torch
from torch.distributions.dirichlet import Dirichlet


class text_processor:
    """
    Clean a corpus with SpaCy.
    """

    def __init__(
        self,
        spacy_model: str,
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: List[str] = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        pos_tags_to_keep: Optional[List[str]] = None,
        remove_chars: Optional[List[str]] = None,
    ):
        """
        Args:
            spacy_model : string, name of the SpaCy model to use
            remove_punctuation : boolean, whether to remove punctuation or not
            remove_digits : boolean, whether to remove digits or not
            stop_words : list of strings, list of stop words to remove
            lowercase : boolean, whether to lowercase or not
            lemmatize : boolean, whether to lemmatize or not
            pos_tags_to_keep : list of strings, list of Part of Speech tags to keep
            remove_chars : list of strings, list of characters to remove
        """

        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)

        self.spacy_model = spacy_model
        self._nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.pos_tags_to_keep = pos_tags_to_keep
        self.remove_chars = remove_chars

    def clean_text(self, s) -> List[str]:
        """
        Clean a SpaCy doc.
        """

        if self.remove_punctuation:
            s = [t for t in s if t.is_punct == False]

        if self.remove_digits:
            s = [t for t in s if t.is_digit == False]

        if self.pos_tags_to_keep is not None:
            temp = []
            for t in s:
                if t.pos_ in self.pos_tags_to_keep:
                    temp.append(t)
            s = temp

        if self.lowercase and not self.lemmatize:
            s = [t.lower_ for t in s]

        if self.lowercase and self.lemmatize:
            s = [t.lemma_.lower() for t in s]

        if not self.lowercase and not self.lemmatize:
            s = [t.text for t in s]

        if self.remove_chars is not None:
            for char in self.remove_chars:
                s = [t.replace(char, " ") for t in s]

        s = [t for t in s if t not in self.stop_words]

        s = [t.strip() for t in s if t not in self.stop_words]

        s = " ".join(s)

        s = s.strip()

        return s

    def process_docs(
        self,
        docs: List[str],
        batch_size: int = 100,
        n_process: int = -1,
        output_path: Optional[str] = None,
        return_docs: bool = False,
        reload_model_every=10000,  # helps with memory leakage issues
        progress_bar: bool = True,
    ):
        """
        SpaCy pipeline with multiprocessing.
        """

        spacy_docs = self._nlp.pipe(docs, batch_size=batch_size, n_process=n_process)

        if progress_bar:
            spacy_docs = tqdm(spacy_docs, total=len(docs))

        counter = 0

        if output_path is None:
            docs_clean = []
            for doc in spacy_docs:
                cleaned_text = self.clean_text(doc)
                docs_clean.append(cleaned_text)
                counter += 1
                if counter > reload_model_every:
                    self._nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
                    counter = 0
        else:
            with open(output_path, "w", newline="") as csvfile:
                fieldnames = ["doc_clean"]
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                for doc in spacy_docs:
                    cleaned_text = self.clean_text(doc)
                    writer.writerow([cleaned_text])
                    counter += 1
                    if counter > reload_model_every:
                        self._nlp = spacy.load(
                            self.spacy_model, disable=["ner", "parser"]
                        )
                        counter = 0
            if return_docs:
                docs_clean = pd.read_csv(output_path)["doc_clean"].tolist()
            else:
                docs_clean = None

        return docs_clean


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            f"the longest document in your collection has {max_local_length} words, the model instead "
            f"truncates to {max_seq_length} tokens."
        )


def bert_embeddings_from_list(
    texts, sbert_model_to_load, batch_size, max_seq_length, device="cpu"
):
    """
    Creates SBERT Embeddings from a list
    """

    model = SentenceTransformer(sbert_model_to_load, device=device)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def compute_mmd_loss(x, y, device, t=0.1, kernel="diffusion"):
    """
    Computes the MMD loss with information diffusion kernel.

    Reference:
        - https://github.com/zll17/Neural_Topic_Models
    """
    eps = 1e-6
    n, d = x.shape
    if kernel == "tv":
        sum_xx = torch.zeros(1).to(device)
        for i in range(n):
            for j in range(i + 1, n):
                sum_xx = sum_xx + torch.norm(x[i] - x[j], p=1).to(device)
        sum_xx = sum_xx / (n * (n - 1))

        sum_yy = torch.zeros(1).to(device)
        for i in range(y.shape[0]):
            for j in range(i + 1, y.shape[0]):
                sum_yy = sum_yy + torch.norm(y[i] - y[j], p=1).to(device)
        sum_yy = sum_yy / (y.shape[0] * (y.shape[0] - 1))

        sum_xy = torch.zeros(1).to(device)
        for i in range(n):
            for j in range(y.shape[0]):
                sum_xy = sum_xy + torch.norm(x[i] - y[j], p=1).to(device)
        sum_yy = sum_yy / (n * y.shape[0])
    else:
        qx = torch.sqrt(torch.clamp(x, eps, 1))
        qy = torch.sqrt(torch.clamp(y, eps, 1))
        xx = torch.matmul(qx, qx.t())
        yy = torch.matmul(qy, qy.t())
        xy = torch.matmul(qx, qy.t())

        def diffusion_kernel(a, tmpt, dim):
            return torch.exp(-torch.acos(a).pow(2)) / tmpt

        off_diag = 1 - torch.eye(n).to(device)
        k_xx = diffusion_kernel(torch.clamp(xx, 0, 1 - eps), t, d - 1)
        k_yy = diffusion_kernel(torch.clamp(yy, 0, 1 - eps), t, d - 1)
        k_xy = diffusion_kernel(torch.clamp(xy, 0, 1 - eps), t, d - 1)
        sum_xx = (k_xx * off_diag).sum() / (n * (n - 1))
        sum_yy = (k_yy * off_diag).sum() / (n * (n - 1))
        sum_xy = 2 * k_xy.sum() / (n * n)

        mmd_loss = sum_xx + sum_yy - sum_xy

    return mmd_loss

def top_k_indices_column(col, k):
    """
    Returns the indices of the top k largest values for each column in a NumPy array.
    """
    return np.argsort(col)[-k:]
