import glob
import logging as log
import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset


class Neuroment2Dataset(Dataset):
    """Neuroment 2 dataset for data loading."""

    def __init__(
        self,
        dataset_dir: str,
        data_type: str,
        standardize_features: bool,
    ):
        """Initialize the Dataset object.

        Args:
            dataset_dir (str): Dataset directory with pickle files.
            data_type (str): Either "training", "validation" or "test".
            standardize_features (str): Whether to standardize features to 0 mean and 1 var or not.
        """
        self.dataset_dir = dataset_dir
        self.data_type = data_type
        self.standardize_features = standardize_features

        if self.standardize_features:
            log.info("Standardizing features to 0 mean and 1 var.")

        if self.data_type not in ["training", "test", "validation"]:
            raise RuntimeError("Data type '%s' is invalid!" % self.data_type)

        self.pickle_files = glob.glob(
            os.path.join(self.dataset_dir, "*%s*.pkl" % self.data_type), recursive=False
        )

        if len(self.pickle_files) == 0:
            raise RuntimeError(
                "Did not find any pickle files with data type '%s' in '%s'!"
                % (self.data_type, self.dataset_dir)
            )

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx):
        # load the Mix object at idx
        with open(self.pickle_files[idx], "rb") as f:
            data = pickle.load(f)

        features = data.feature_mix
        if len(features.shape) == 2:
            # add channel dimension
            features = features[np.newaxis, ...]
        labels = data.labels_mix

        if self.standardize_features:
            features = standardize_features(features)

        return features.astype("float32"), labels.astype("float32")


def standardize_features(features: np.ndarray):
    """Standardizes features to 0 mean and 1 var accross time dimension."""
    features_scaled = np.zeros(features.shape)

    for channel in range(features.shape[0]):
        # compute mean and standard deviation over time dimension
        mean = np.mean(features[channel, :, :], axis=-1)
        std = np.std(features[channel, :, :], axis=-1)

        # scale features
        features_scaled[channel, :, :] = (
            features[channel, :, :] - mean[:, np.newaxis]
        ) / (std[:, np.newaxis] + 1e-12)

    return features_scaled
