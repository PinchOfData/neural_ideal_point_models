# neural_ideal_point_models

A pytorch implementation of the Neural Ideal Point Model (IdealPointNN).

## Brief Description

The Neural Ideal Point Model rely on a Wasserstein autoencoder to approximate ideal points conditional on an observed dataset and a researcher prior. It is particularly well-suited for unstructured data (e.g., texts, images).

The Neural Ideal Point Model present various advantages:
- **Many supported modalities**: The model takes voting records, survey responses, word frequencies, and/or embeddings as input.
- **Multimodality**: Users can jointly learn ideal points from different modalities (e.g., votes *and* word frequencies).
- **Soft supervision**: Users can softly enforce ideal points to be predictive of an outcome (e.g., partisanship).
- **Conditioned on covariates**: Users can augment the model with additional covariates. 

## References

The Neural Ideal Point Model, Germain Gauthier and Hugo Subtil, *coming soon*

## Disclaimer

The package is usable but still in development :)