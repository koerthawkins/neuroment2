import glob
import os

import numpy as np
import librosa as lb

from neuroment2.utils import delete_by_indices


class Mixer:
    """Class handling the mixing of audio samples"""

    def __init__(
        self,
        num_epochs=None,
        data_path=None,
        num_instruments=None,
        dataset=None,
        **kwargs,
    ):
        self.cfg_mixes = kwargs["Mix"]

        self.num_epochs = num_epochs
        self.data_path = data_path
        self.dataset = dataset
        assert (
            len(num_instruments) == 2
        ), "boundaries of number of instruments must have a length of 2"
        self.min_num_instruments = num_instruments[0]
        self.max_num_instruments = num_instruments[1]
        self.create_file_list()

        self.mixes = []
        self.create_mixes("training")
        pass

    def create_file_list(self):
        os.chdir(self.data_path + self.dataset)
        files = {}
        files["training"] = []
        files["test"] = []
        files["validation"] = []
        for file in glob.glob("*.wav"):
            if "training" in file:
                files["training"].append(file)
            elif "test" in file:
                files["test"].append(file)
            elif "validation" in file:
                files["validation"].append(file)
        self.file_list = files

    def create_mixes(self, type_of_dataset):
        file_list_temp = self.file_list[type_of_dataset].copy()
        while len(file_list_temp) >= self.max_num_instruments:
            num_files_left = len(file_list_temp)
            num_instruments = np.random.randint(
                self.min_num_instruments, self.max_num_instruments
            )
            indices_file_list = np.random.randint(
                0, num_files_left - 1, size=num_instruments
            )
            files_mix = []
            for index in indices_file_list:
                files_mix.append(file_list_temp[index])
            delete_by_indices(file_list_temp, indices_file_list)
            # self.mixes.append(Mix(files_mix, self.cfg_mixes))
        print()


# class Mix:
#     """Container holding mixed audio data and its features."""

#     def __init__(self, files_mix, **kwargs):
#         self.sr = kwargs["sr"]
#         self.feature = kwargs["feature"]
#         self.num_files = len(files_mix)
#         self.files_mix = files_mix
#         self.calculate_feature()
#         pass

#     def calculate_feature(self):
#         wav_files = []
#         for file in self.files_mix:
#             wav_files.append(lb.load(file, self.sr))
