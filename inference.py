import glob
import hydra
import logging as log
import librosa as lb
from omegaconf import DictConfig
import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from neuroment2 import *


INSTRUMENT_LIST = ["clarinet", "e-guitar", "singer", "flute", "piano", "saxophone", "trumpet", "violin"]


@hydra.main(config_path="conf", config_name="config")
def inference(cfg: DictConfig) -> None:
    # we switch to repository root dir to simplify path handling
    # (inference doesn't have intermediate results, so no need to introduce an extra hydra dir)
    os.chdir(hydra.utils.to_absolute_path("."))

    # use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(cfg.train.gpu_index))
    else:
        device = "cpu"

    log.info("Selected device: %s" % device)

    # load state dict s.t. we get parameters to create model
    state_dict_model = torch.load(cfg.inference.model_path, map_location=device)
    dataset_stats = state_dict_model["dataset_stats"]
    log.info("Loaded model: '%s'." % cfg.inference.model_path)

    # create model
    model = NeuromentModel(
        num_instruments=dataset_stats["num_instruments"],
        num_input_features=dataset_stats["num_features_per_observation"],
        num_input_frames=dataset_stats["num_frames_per_observation"],
        use_batch_norm=state_dict_model["use_batch_norm"],
        dropout_rate=state_dict_model["dropout_rate"],
    )
    model.load_state_dict(state_dict_model["model"])
    model.to(device)

    # create feature generator
    feature_gen = FeatureGenerator(dataset_stats["feature_generator_cfg"])

    # create predictions directory
    os.makedirs(cfg.inference.predictions_dir, exist_ok=True)
    log.info("Saving predictions to: %s" % os.path.abspath(cfg.inference.predictions_dir))

    # set model to evaluation mode
    model.eval()

    # get list of audio files
    inference_file_list = []
    for ending in ["wav", "flac", "mp3", "WAV", "FLAC", "MP3"]:
        inference_file_list.extend(glob.glob(os.path.join(cfg.inference.audio_dir, "*." + ending)))

    # make progbar
    prog_bar = tqdm(total=len(inference_file_list), desc="Predicting...")

    # compute hann window which we need to overlap predicted frames and check if it's conditioned correctly
    num_frames_per_observation = dataset_stats["num_frames_per_observation"]
    hann_window = lb.filters.get_window("hann", num_frames_per_observation, fftbins=True)
    overlapped_test_window = hann_window[:len(hann_window) // 2] + hann_window[len(hann_window) // 2:]
    assert np.all(np.abs(overlapped_test_window - 1.0) < 1e-3)

    with torch.no_grad():
        for input_file in inference_file_list:
            # load raw audio
            audio, _ = lb.load(input_file, sr=dataset_stats["sr"], mono=True,)

            # generate features
            features, envelope_ref = feature_gen.generate(audio)

            # convert to pytorch tensor and add batch and channel dimensions
            features = torch.Tensor(features).unsqueeze(0).unsqueeze(0)

            # pre-allocate predictions
            n_frames_total = features.shape[-1]
            envelope_pred = np.zeros([dataset_stats["num_instruments"], n_frames_total])

            # now predict and overlap each predicted window by its half using a hann window for perfect overlap
            # we simply skip the last 2 frames in order to avoid an out-of-index error
            # TODO zero-pad instead
            start_indices = np.arange(0, n_frames_total, num_frames_per_observation // 2)
            for start_index in start_indices[:-2]:
                # predict
                pred = model(features[:, :, :, start_index:start_index + num_frames_per_observation])

                # convert predicted envelope from torch Tensor to np.ndarray and remove batch dimension
                pred = pred.cpu().numpy()[0, :, :]

                # multiply with hann window for overlap
                pred = np.multiply(pred, hann_window)

                # add predicted and windowed envelope to collected envelope
                envelope_pred[:, start_index:start_index + num_frames_per_observation] += pred

            # plot results
            plt.subplots(1, 1, figsize=(12, 6.75))

            plt.subplot(1, 1, 1)
            envelope_pred_log = 20 * np.log10(envelope_pred + 1e-12)
            plt.imshow(envelope_pred_log, origin="lower", aspect="auto", vmin=-100, vmax=0)
            plt.colorbar(label="amplitude / dB")

            plt.yticks(ticks=np.arange(0, len(INSTRUMENT_LIST)), labels=INSTRUMENT_LIST)
            plt.title(os.path.basename(input_file))
            plt.xlabel("frames")
            plt.ylabel("instruments")

            plt.tight_layout()
            plt.show()

            # update progress bar
            prog_bar.update(1)


if __name__ == "__main__":
    inference()
