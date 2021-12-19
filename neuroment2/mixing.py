import glob
import os

import numpy as np
from librosa.core import load

from neuroment2.utils import delete_by_indices


class Mixer:
    """Class handling the mixing of audio samples"""

    def __init__(
        self,
        num_epochs=None,
        data_path=None,
        num_instruments=None,
        dataset=None,
        num_samples_per_file=None,
        **kwargs,
    ):
        self.cfg_mixes = kwargs["Mix"]
        self.num_samples_per_file = num_samples_per_file
        self.num_observation_windows = self.get_num_observation_windows()

        self.num_epochs = num_epochs
        self.data_path = data_path
        self.dataset = dataset
        assert (
            len(num_instruments) == 2
        ), "boundaries of number of instruments must have a length of 2"
        self.min_num_instruments = num_instruments[0]
        self.max_num_instruments = num_instruments[1]
        self.create_file_list("training")

        self.mix_id = 0
        self.mixes = []
        self.create_mixes()
        pass

    def create_file_list(self, type):
        os.chdir(self.data_path + self.dataset)
        files = []
        for file in glob.glob("*.wav"):
            if type in file:
                files.append(file)
        num_files = len(files)
        files_observation_windows = []
        for f in range(num_files):
            for i in range(self.num_observation_windows):
                files_observation_windows.append([files[f], i])
        self.file_list = files_observation_windows

    def create_mixes(self):
        for epoch in range(self.num_epochs):
            file_list_temp = self.file_list.copy()
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
                self.mixes.append(Mix(files_mix, self.mix_id, **self.cfg_mixes))
                self.mix_id += 1
        print()

    def get_num_observation_windows(self):
        dft_size = self.cfg_mixes["dft_size"]
        hopsize = self.cfg_mixes["hopsize"]
        num_frames = self.cfg_mixes["num_frames"]
        return int(
            np.round(
                self.num_samples_per_file / (dft_size + (num_frames - 1) * hopsize)
            )
        )


class Mix:
    """Container holding mixed audio data and its features."""

    def __init__(
        self,
        files_mix,
        mix_id,
        sr=None,
        feature=None,
        dft_size=None,
        hopsize=None,
        num_frames=None,
        **kwargs,
    ):
        self.sr = sr
        self.feature = feature
        self.dft_size = dft_size
        self.hopsize = hopsize
        self.num_frames = num_frames

        self.mix_id = mix_id
        self.num_files = len(files_mix)
        self.files_mix = files_mix
        self.calculate_feature()
        pass

    def calculate_feature(self):
        duration = (self.dft_size + (self.num_frames - 1) * self.hopsize) / self.sr
        # offset = [duration * o for _, o in self.files_mix]
        wav_files = []
        for file, offset in self.files_mix:
            audio_data, _ = load(
                file, self.sr, offset=(duration * offset), duration=duration
            )
            wav_files.append(audio_data)

    def generate_labels(self):
        pass
