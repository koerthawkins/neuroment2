import glob
import hydra
import logging as log
import librosa as lb
from omegaconf import DictConfig
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from neuroment2 import *


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

    with torch.no_grad():
        for input_file in inference_file_list:
            # load raw audio
            audio, _ = lb.load(input_file, sr=dataset_stats["sr"], mono=True,)

            # generate features
            features, envelope_ref = feature_gen.generate(audio)

            # convert to pytorch tensor and add batch and channel dimensions
            features = torch.Tensor(features).unsqueeze(0).unsqueeze(0)

            # predict
            envelope_pred = model(features[:, :, :, :14])

            # create new progress bar
            prog_bar.update(1)


if __name__ == "__main__":
    inference()
