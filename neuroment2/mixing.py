import glob
import os
import pickle as pk
import yaml
import logging as log
import numpy as np
from librosa.core import audio, load
import librosa as lb

from neuroment2.utils import delete_by_indices
import matplotlib.pyplot as plt


class FeatureGenerator:
    def __init__(
        self, cfg
    ):  # feature:str, sr:int, dft_size:int, hopsize:int, num_mel, window):
        self.cfg = cfg

        self.feature = cfg["feature"]
        self.sr = cfg["Mix"]["sr"]
        self.dft_size = cfg["Mix"]["dft_size"]
        self.hopsize = cfg["Mix"]["hopsize"]
        self.window = cfg["window"]
        self.num_mels = cfg["num_mels"]
        self.f_min = cfg["f_min"]
        self.f_max = cfg["f_max"]
        self.center = cfg["center"]
        self.num_bins_per_octave = cfg["num_bins_per_octave"]
        self.num_octaves = cfg["num_octaves"]

    def generate(self, audio):
        if self.feature == "CQT":
            cqt = lb.cqt(
                audio,
                hop_length=self.hopsize,
                n_bins=self.num_bins_per_octave * self.num_octaves,
                bins_per_octave=self.num_bins_per_octave,
                fmin=self.f_min,
                window=self.window,
                scale=False,
            )
            cqt = np.abs(cqt)

            cqt_lengths = lb.filters.constant_q_lengths(
                sr=self.sr,
                n_bins=self.num_bins_per_octave * self.num_octaves,
                bins_per_octave=self.num_bins_per_octave,
                fmin=self.f_min,
                window=self.window,
            )
            cqt /= cqt_lengths[:, None]
            envelope = np.sum((cqt ** 2.0), axis=0)
            feature = np.log(cqt + 1e-12)
        elif self.feature == "STFT":
            stft = np.abs(
                lb.core.stft(
                    audio,
                    n_fft=self.dft_size,
                    hop_length=self.hopsize,
                    window=self.window,
                    center=self.center,
                )
            )
            stft /= self.dft_size
            envelope = np.sum(stft ** 2.0, axis=0)
            feature = np.log(stft + 1e-12)
        elif self.feature == "MEL":
            stft = np.abs(
                lb.core.stft(
                    audio,
                    n_fft=self.dft_size,
                    hop_length=self.hopsize,
                    window=self.window,
                    center=self.center,
                )
            )
            stft /= self.dft_size
            mel_filter_bank = lb.filters.mel(
                sr=self.sr,
                n_fft=self.dft_size,
                n_mels=self.num_mels,
                fmin=self.f_min,
                fmax=self.f_max,
                norm=1.0,
            )
            mel = np.dot(mel_filter_bank, stft)
            envelope = np.sum(mel ** 2.0, axis=0)
            feature = np.log(mel + 1e-12)
        else:
            raise KeyError("feature type not available")

        return feature, envelope


class Mixer:
    """Class handling the mixing of audio samples"""

    def __init__(
        self,
        num_epochs=None,
        pickle_path=None,
        num_instruments_mix=None,
        num_instruments=None,
        raw_data_path=None,
        data_type=None,
        num_samples_per_file=None,
        **kwargs,
    ):
        self.cfg = kwargs
        self.cfg_mixes = kwargs["Mix"]
        self.num_samples_per_file = num_samples_per_file
        self.num_observation_windows = self.get_num_observation_windows()

        self.feature_generator = FeatureGenerator(kwargs)

        self.num_epochs = num_epochs
        self.pickle_path = pickle_path
        self.raw_data_path = raw_data_path
        self.data_type = data_type
        assert (
            len(num_instruments_mix) == 2
        ), "boundaries of number of instruments must have a length of 2"
        self.min_num_instruments = num_instruments_mix[0]
        self.max_num_instruments = num_instruments_mix[1]
        self.num_instruments = num_instruments
        self.create_file_list(self.data_type)

        self.mix_id = 0
        self.create_mixes()
        pass

    def create_file_list(self, type: str):
        """creates a list of files for the selected category

        Args:
            type (str): type of data
        """
        files = []

        # get raw audio files
        for file in glob.glob(os.path.join(self.raw_data_path, "*.wav")):
            if type in file:
                files.append(file)

        # for each audio file we generate a list with all possible observation windows
        files_observation_windows = []
        for f in range(len(files)):
            for i in range(self.num_observation_windows):
                files_observation_windows.append([files[f], i])

        self.file_list = files_observation_windows

    def create_mixes(self):
        # create output directory
        os.makedirs(self.pickle_path, exist_ok=True)

        # loop through dataset epochs
        for _ in range(self.num_epochs):
            # we create a copy of the original file list where we pop files from
            file_list_temp = self.file_list.copy()

            # generate files until the max number of instruments can not be satisfied anymore
            while len(file_list_temp) >= self.max_num_instruments:
                num_instruments_mix = np.random.randint(
                    self.min_num_instruments, self.max_num_instruments + 1
                )
                indices_file_list = np.random.randint(
                    0, len(file_list_temp), size=num_instruments_mix
                )

                files_mix = []
                for index in indices_file_list:
                    files_mix.append(file_list_temp[index])
                delete_by_indices(file_list_temp, indices_file_list)

                # generate the mix
                mix = Mix(
                    files_mix,
                    self.num_instruments,
                    self.feature_generator,
                    self.mix_id,
                    self.raw_data_path,
                    **self.cfg_mixes,
                )

                # write the generated mix to a pickle file
                pickle_path = os.path.join(self.pickle_path, f"{self.data_type}_mix_{self.mix_id}.pkl")
                with open(pickle_path, "wb") as f:
                    pk.dump(mix, f)
                log.info("Wrote '%s'." % pickle_path)

                self.mix_id += 1

        # save statistics to a separate YAML file s.t. we can import the metadata for training
        self.save_statistics()

    def save_statistics(self):
        statistics = {}
        statistics["num_pickle"] = self.mix_id
        statistics["num_instruments"] = self.num_instruments
        if self.cfg["center"]:
            statistics["num_frames_per_observation"] = self.cfg_mixes["num_frames"] + 2
        else:
            statistics["num_frames_per_observation"] = self.cfg_mixes["num_frames"]

        if self.cfg["feature"] == "CQT":
            statistics["num_features_per_observation"] = (
                self.cfg["num_octaves"] * self.cfg["num_bins_per_octave"]
            )
        elif self.cfg["feature"] == "STFT":
            statistics["num_features_per_observation"] = self.dft_size // 2 + 1
        elif self.cfg["feature"] == "MEL":
            statistics["num_features_per_observation"] = self.cfg["num_mels"]

        statistics["feature"] = self.cfg["feature"]
        statistics["dft_size"] = self.cfg_mixes["dft_size"]
        statistics["hopsize"] = self.cfg_mixes["hopsize"]
        statistics["sr"] = self.cfg_mixes["sr"]

        statistics["feature_generator_cfg"] = self.feature_generator.cfg

        statistics_path = os.path.join(self.pickle_path, "%s_statistics.yml" % self.data_type)
        with open(statistics_path, "w") as outfile:
            yaml.dump(statistics, outfile, default_flow_style=False)
        log.info("Wrote '%s'." % statistics_path)

    def get_num_observation_windows(self):
        dft_size = self.cfg_mixes["dft_size"]
        hopsize = self.cfg_mixes["hopsize"]
        num_frames = self.cfg_mixes["num_frames"]
        return int(
            np.floor(
                self.num_samples_per_file / (dft_size + (num_frames - 1) * hopsize)
            )
        )


class Mix:
    """Container holding mixed audio data and its features."""

    def __init__(
        self,
        files_mix,
        num_instruments,
        feature_generator,
        mix_id,
        dataset,
        sr=None,
        feature=None,
        dft_size=None,
        hopsize=None,
        num_frames=None,
        save_wav_data=None,
        **kwargs,
    ):
        self.sr = sr
        self.feature = feature
        self.dft_size = dft_size
        self.hopsize = hopsize
        self.num_frames = num_frames

        self.mix_id = mix_id
        self.dataset = dataset
        self.num_files = len(files_mix)
        self.files_mix = files_mix
        self.num_instruments = num_instruments

        self.wav_files = []
        self.features = []
        self.envelopes = []
        self.label_list = []

        self.feature_mix, self.labels_mix = self.calculate_feature(
            feature_generator, save_wav_data
        )

    def calculate_feature(self, feature_generator, save_wav_data):
        # TODO Try to fix +1 offset
        duration = (
            (self.dft_size + (self.num_frames - 1) * self.hopsize) + 1
        ) / self.sr

        levels = np.clip(np.random.normal(0.5, 0.1, self.num_files), 0.0, 1.0)
        levels /= np.sum(levels)
        for i, (file, offset) in enumerate(self.files_mix):
            audio_data, _ = load(
                file, self.sr, offset=(duration * offset), duration=duration
            )
            audio_data *= levels[i]
            feature, envelope = feature_generator.generate(audio_data)
            label = int(file.split("-")[1][0])

            self.wav_files.append(audio_data)
            self.features.append(feature)
            self.envelopes.append(envelope)
            self.label_list.append(label)

        audio_mix = np.sum(np.stack(self.wav_files) * levels[:, np.newaxis], axis=0)
        feature_mix, _ = feature_generator.generate(audio_mix)
        labels_mix = np.zeros((self.num_instruments, feature_mix.shape[1]))
        for i in range(self.num_files):
            labels_mix[self.label_list[i], :] = self.envelopes[i]
        if not save_wav_data:
            self.wav_files = []
        return feature_mix, labels_mix
