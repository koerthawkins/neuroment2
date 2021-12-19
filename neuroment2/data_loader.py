import glob
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
        dataset_dir,
        data_type,
    ):
        """Initialize the Dataset object.

        Args:
            dataset_dir (str): Dataset directory with pickle files.
            data_type (str): Either "training", "validation" or "test".
        """
        self.dataset_dir = dataset_dir
        self.data_type = data_type

        if self.data_type not in ["training", "test", "validation"]:
            raise RuntimeError("Data type '%s' is invalid!" % self.data_type)

        self.pickle_files = glob.glob(os.path.join(self.dataset_dir, "*%s*.pkl" % self.data_type), recursive=False)

        if len(self.pickle_files) == 0:
            raise RuntimeError("Did not find any pickle files with data type '%s' in '%s'!"
                               % (self.data_type, self.dataset_dir))

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx):
        # load the Mix object at idx
        with open(self.pickle_files[idx], "rb") as f:
            data = pickle.load(f)

        features = data.feature_mix
        labels = data.labels_mix

        return features.astype("float32"), labels.astype("float32")
