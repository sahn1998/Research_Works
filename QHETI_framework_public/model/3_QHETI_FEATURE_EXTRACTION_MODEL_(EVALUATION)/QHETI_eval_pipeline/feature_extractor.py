"""
FeatureExtractor Module

This module defines a class to extract intermediate CNN features in batches using a
pretrained Keras model. It is typically used to convert structured CNN image outputs
into flattened feature vectors that can be fed into conventional machine learning models.
"""
import numpy as np

class FeatureExtractor:
    """
    Extracts features from a Keras model (e.g., penultimate CNN layer) in batches.
    """

    def __init__(self, feature_extractor_model, batch_size):
        """
        Args:
            model (keras.Model): A Keras model with output from a hidden layer.
            batch_size (int): Number of samples to process in one batch.
        """
        self.feature_extractor_model = feature_extractor_model
        self.batch_size = batch_size

    def extract(self, data):
        """
        Extracts and flattens features from input data.

        Args:
            data (np.ndarray): Input images of shape (N, H, W, C).

        Returns:
            np.ndarray: Flattened feature matrix of shape (N, D).
        """
        num_samples = data.shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        feature_batches = [
            self.feature_extractor_model.predict(data[i * self.batch_size: min((i + 1) * self.batch_size, num_samples)], verbose=0)
            for i in range(num_batches)
        ]

        stacked_features = np.vstack(feature_batches)
        return stacked_features.reshape(stacked_features.shape[0], -1)