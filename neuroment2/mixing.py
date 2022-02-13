import glob
import os
import pickle as pk
import yaml
import logging as log
import numpy as np
from librosa.core import audio, load
import librosa as lb
from tqdm import tqdm

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
        self.envelope_type = cfg["envelope_type"]

        if self.envelope_type not in ["RMS", "SPECTRUM"]:
            raise ValueError("envelope_type '%s' is invalid!" % self.envelope_type)

        # generate some filter bank variables which don't need to be re-computed each time
        # in self.generate()
        self.cqt_lengths = lb.filters.constant_q_lengths(
                sr=self.sr,
                n_bins=self.num_bins_per_octave * self.num_octaves,
                bins_per_octave=self.num_bins_per_octave,
                fmin=self.f_min,
                window=self.window,
        )
        # filter bank for standalone mel features
        self.mel_filter_bank = lb.filters.mel(
                sr=self.sr,
                n_fft=self.dft_size,
                n_mels=self.num_mels,
                fmin=self.f_min,
                fmax=self.f_max,
                norm=1.0,
        )
        # filter bank for CQT+MEL features
        self.mel_filter_bank_combo_features = lb.filters.mel(
                sr=self.sr,
                n_fft=self.dft_size,
                n_mels=len(self.cqt_lengths),
                fmin=self.f_min,
                fmax=self.f_max,
                norm=1.0,
        )

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

            cqt /= self.cqt_lengths[:, None]
            feature = np.log(cqt + 1e-12)

            if self.envelope_type == "SPECTRUM":
                envelope = np.sum((cqt ** 2.0), axis=0)
            else:
                envelope = self._compute_rms_envelope(audio)
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
            feature = np.log(stft + 1e-12)

            if self.envelope_type == "SPECTRUM":
                envelope = np.sum(stft ** 2.0, axis=0)
            else:
                envelope = self._compute_rms_envelope(audio)
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

            mel = np.dot(self.mel_filter_bank, stft)
            feature = np.log(mel + 1e-12)

            if self.envelope_type == "SPECTRUM":
                envelope = np.sum(mel ** 2.0, axis=0)
            else:
                envelope = self._compute_rms_envelope(audio)
        elif self.feature == "CQT+MEL":
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

            cqt /= self.cqt_lengths[:, None]
            cqt = np.log(cqt + 1e-12)

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

            mel = np.dot(self.mel_filter_bank_combo_features, stft)
            mel = np.log(mel + 1e-12)

            # pack CQT into first and MEL into second channel
            feature = np.zeros(shape=[2] + list(cqt.shape))
            feature[0, :, :] += cqt
            feature[1, :, :] += mel

            # here only RMS envelope makes sense
            envelope = self._compute_rms_envelope(audio)

        else:
            raise KeyError("feature type not available")

        # add channel dimension to features if they aren't in features yet
        if len(feature.shape) == 2:
            assert feature.shape[0] != 1
            feature = feature[np.newaxis, ...]

        return feature, envelope

    def _compute_rms_envelope(self, audio: np.ndarray):
        """Computes the RMS envelope in time domain.
        Args:
            audio: audio vector, mono, 1D

        Returns:
            rms: RMS values (linear, no dB)
        """
        rms = lb.feature.rms(audio, frame_length=self.dft_size, hop_length=self.hopsize, center=self.center)[0, :]

        return rms


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

        self.num_epochs = num_epochs[data_type]
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

    def create_file_list(self, data_type: str):
        """creates a list of files for the selected category

        Args:
            type (str): type of data
        """
        files = []

        # get raw audio files
        for file in glob.glob(os.path.join(self.raw_data_path, "*.wav")):
            if data_type in file:
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

        # compute number of mixes to generate
        n_mixes = self.num_epochs * len(self.file_list)
        n_mixes /= np.mean([self.min_num_instruments, self.max_num_instruments])
        n_mixes = int(n_mixes)

        # init progress bar
        progress_bar = tqdm(total=n_mixes)
        progress_bar.set_description(self.data_type)

        # init mix counter
        i_mix = 0

        # loop through dataset epochs
        while i_mix < n_mixes:
            # we create a copy of the original file list where we pop files from
            file_list_temp = self.file_list.copy()

            # we also shuffle it to make sure we don't use the same order twice
            np.random.shuffle(file_list_temp)

            # generate files until the max number of instruments can not be satisfied anymore
            while len(file_list_temp) >= self.max_num_instruments:
                # init counter at beginning
                i_mix += 1

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
                try:
                    mix = Mix(
                        files_mix,
                        self.num_instruments,
                        self.feature_generator,
                        self.mix_id,
                        self.raw_data_path,
                        **self.cfg_mixes,
                    )

                    # remove all redundant fields from mix s.t. we don't need too much storage
                    del mix.envelopes
                    del mix.features
                    del mix.wav_files
                    mix.feature_mix = mix.feature_mix.astype(np.float32)
                    mix.labels_mix = mix.labels_mix.astype(np.float32)

                    # write the generated mix to a pickle file
                    pickle_path = os.path.join(self.pickle_path, f"{self.data_type}_mix_{self.mix_id}.pkl")
                    with open(pickle_path, "wb") as f:
                        pk.dump(mix, f)
                    # log.info("Wrote '%s'." % pickle_path)
                except Exception as e:
                    # in some really rare cases the audio can get non-finite
                    log.warning("Could not generate mix due to exception '%s'!" % str(e))

                self.mix_id += 1
                progress_bar.update(1)

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

        if self.cfg["feature"] in ["CQT", "CQT+MEL"]:
            statistics["num_features_per_observation"] = (
                self.cfg["num_octaves"] * self.cfg["num_bins_per_octave"]
            )
        elif self.cfg["feature"] == "STFT":
            statistics["num_features_per_observation"] = self.cfg["Mix"]["dft_size"] // 2 + 1
        elif self.cfg["feature"] == "MEL":
            statistics["num_features_per_observation"] = self.cfg["num_mels"]

        if self.cfg["feature"] == "CQT+MEL":
            statistics["num_channels"] = 2
        else:
            statistics["num_channels"] = 1

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

            # get base file name without directory
            file_name = os.path.basename(file)

            # get file parts
            parts = file_name.split("_")
            dataset_label_and_label = parts[1]

            # derive label from file name
            label = int(dataset_label_and_label.split("-")[1])

            self.wav_files.append(audio_data)
            self.features.append(feature)
            self.envelopes.append(envelope)
            self.label_list.append(label)

        audio_mix = np.sum(np.stack(self.wav_files) * levels[:, np.newaxis], axis=0)
        feature_mix, _ = feature_generator.generate(audio_mix)
        labels_mix = np.zeros([self.num_instruments, feature_mix.shape[-1]])
        for i in range(self.num_files):
            labels_mix[self.label_list[i], :] = self.envelopes[i]
        if not save_wav_data:
            self.wav_files = []
        return feature_mix, labels_mix
