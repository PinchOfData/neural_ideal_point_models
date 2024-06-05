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