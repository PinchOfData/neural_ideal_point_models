import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderMLP(nn.Module):
    """
    Torch implementation of an encoder Multilayer Perceptron.
    """
    def __init__(
        self,
        encoder_dims=[2000, 1024, 512, 20],
        encoder_non_linear_activation="relu",
        encoder_bias=True,
        dropout=0.0,
    ):
        super(EncoderMLP, self).__init__()

        self.encoder_dims = encoder_dims
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.dropout = nn.Dropout(p=dropout)
        if encoder_non_linear_activation is not None:
            self.encoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                encoder_non_linear_activation
            ]

        self.encoder = nn.ModuleDict(
            {
                f"enc_{i}": nn.Linear(
                    encoder_dims[i], encoder_dims[i + 1], bias=encoder_bias
                )
                for i in range(len(encoder_dims) - 1)
            }
        )

    def forward(self, x):
        """
        Encode the input.
        """
        hid = x
        for i, (_, layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if (
                i < len(self.encoder) - 1
                and self.encoder_non_linear_activation is not None
            ):
                hid = self.encoder_nonlin(hid)
        return hid


class DecoderMLP(nn.Module):
    """
    Torch implementation of a decoder Multilayer Perceptron.
    """
    def __init__(
        self,
        decoder_dims=[20, 1024, 2000],
        decoder_non_linear_activation=None,
        decoder_bias=False,
        dropout=0.0,
    ):
        super(DecoderMLP, self).__init__()

        self.decoder_dims = decoder_dims
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.dropout = nn.Dropout(p=dropout)
        if decoder_non_linear_activation is not None:
            self.decoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                decoder_non_linear_activation
            ]

        self.decoder = nn.ModuleDict(
            {
                f"dec_{i}": nn.Linear(
                    decoder_dims[i], decoder_dims[i + 1], bias=decoder_bias
                )
                for i in range(len(decoder_dims) - 1)
            }
        )

    def forward(self, z):
        """
        Decode the input.
        """
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = self.dropout(layer(hid))
            if (
                i < len(self.decoder) - 1
                and self.decoder_non_linear_activation is not None
            ):
                hid = self.decoder_nonlin(hid)
        return hid
    

class DecoderTBIP(nn.Module):
    def __init__(self, input_dim, n_topics, vocab_size, dropout=0.0):
        """
        Decoder for computing Poisson rate based on topic shares, ideal points, 
        and ideological adjustments.
        
        Args:
        - input_dim: Dimension of the ideal points (z).
        - n_topics: Number of topics (dimension of θ).
        - vocab_size: Vocabulary size (dimension of β and η).
        - dropout: Dropout rate for regularization.
        """
        super(DecoderTBIP, self).__init__()
        
        # Layer for generating β (neutral topic intensities) for each word per topic
        self.topic_to_beta = nn.Parameter(torch.randn(n_topics, vocab_size))
        
        # Layer for generating η (ideological adjustments) for each word per topic
        self.ideal_to_eta = nn.Parameter(torch.randn(input_dim, vocab_size))
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, theta, z):
        """
        Forward pass to compute the Poisson rate λ for each document.
        
        Args:
        - theta: Topic shares for each document (batch_size, n_topics).
        - z: Ideal points for each document (batch_size, input_dim).
        
        Returns:
        - lambda_: Poisson rate for each word in the vocabulary (batch_size, vocab_size).
        """
        # Compute neutral topic term: θ * β
        beta_term = torch.matmul(theta, self.topic_to_beta)  # Shape: (batch_size, vocab_size)
        
        # Compute ideological term: exp(z * η)
        eta_term = torch.matmul(z, self.ideal_to_eta)  # Shape: (batch_size, vocab_size)
        eta_term = torch.clamp(eta_term, max=5)  # Clip to prevent overflow
        exp_eta_term = torch.exp(eta_term)  # Exponentiate to get positive adjustments
        
        # Compute the Poisson rate λ as element-wise product of θβ and exp(zη)
        lambda_ = beta_term * exp_eta_term + 1e-8  # Shape: (batch_size, vocab_size)
        
        #print("Max value in eta_term:", eta_term.max().item())
        #print("Max value in beta_term:", beta_term.max().item())
        #print("Max value in lambda_:", lambda_.max().item())
        #assert not torch.isnan(lambda_).any(), "NaN in lambda_"  # Check for NaNs

        return lambda_
