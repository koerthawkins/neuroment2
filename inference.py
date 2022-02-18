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
import seaborn as sns

from neuroment2 import *


@hydra.main(config_path="conf", config_name="config")
def inference(cfg: DictConfig) -> None:
    # we switch to repository root dir to simplify path handling
    # (inference doesn't have intermediate results, so no need to introduce an extra hydra dir)
    os.chdir(hydra.utils.to_absolute_path("."))

    # use GPU if available
    if cfg.inference.gpu_index != -1 and torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(cfg.inference.gpu_index))
    else:
        device = "cpu"

    log.info("Selected device: %s" % device)

    # load state dict s.t. we get parameters to create model
    state_dict_model = torch.load(cfg.inference.model_path, map_location=device)
    dataset_stats = state_dict_model["dataset_stats"]
    log.info("Loaded model: '%s'." % cfg.inference.model_path)

    # create model
    model = NeuromentModel(
        num_channels=dataset_stats.get("num_channels", 1),  # small compatibility fix here
        num_instruments=dataset_stats["num_instruments"],
        num_input_features=dataset_stats["num_features_per_observation"],
        num_input_frames=dataset_stats["num_frames_per_observation"],
        use_batch_norm=state_dict_model["use_batch_norm"],
        dropout_rate=state_dict_model["dropout_rate"],
    )
    model.load_state_dict(state_dict_model["model"])
    model = model.to(device)

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

            # add channel dimension
            if len(features.shape) == 2:
                features = features[np.newaxis, ...]

            # standardize features if model was trained with standardized features
            if state_dict_model.get("standardize_features", False):
                features = standardize_features(features)

            # convert to pytorch tensor and add batch dimensions
            features = torch.Tensor(features).unsqueeze(0)

            # move features to device
            features = features.to(device)

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
            fig, _ = plt.subplots(1, 1, figsize=cfg.figsize)

            plt.subplot(1, 1, 1)

            # imshow() automatically interpolates, so we use matshow()
            plt.matshow(to_db(envelope_pred), fignum=0, aspect="auto", origin="lower", vmin=-100, vmax=0)

            plt.colorbar(label="amplitude / dB")

            t = np.linspace(0.0, len(audio) / float(dataset_stats["sr"]), num=n_frames_total)
            t_string = ["%.1f" % number for number in t]
            step_width = n_frames_total // 8
            plt.xticks(ticks=np.arange(0, n_frames_total)[::step_width], labels=t_string[::step_width])
            plt.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            plt.yticks(ticks=np.arange(0, len(cfg.class_labels)), labels=cfg.class_labels)
            plt.title(os.path.basename(input_file))
            plt.xlabel("time / s")
            plt.ylabel("instruments")

            plt.tight_layout()
            plt.show()

            # save plot to file
            plot_file_path = os.path.join(
                cfg.inference.predictions_dir,
                os.path.splitext(os.path.basename(input_file))[0] + ".png",
            )
            fig.savefig(plot_file_path)
            log.info("Saved '%s'." % plot_file_path)

            if "sequence" in input_file.lower():
                envelope_ref = compute_reference_envelopes(
                    cfg,
                    audio,
                    dataset_stats["feature_generator_cfg"],
                    n_frames_total,
                    sample_length_per_instrument=5.0,
                    total_sample_length=len(audio) / float(dataset_stats["sr"]),
                )

                # compute some plot variables
                t = np.linspace(0.0, len(audio) / float(dataset_stats["sr"]), num=n_frames_total)
                t_string = ["%.1f" % number for number in t]
                step_width = n_frames_total // 8

                # make comparison plot
                fig, _ = plt.subplots(2, 1, figsize=cfg.figsize)

                # plot predicted envelopes
                plt.subplot(2, 1, 1)
                plt.matshow(to_db(envelope_pred), fignum=0, aspect="auto", origin="lower", vmin=-100, vmax=0)
                plt.colorbar(label="amplitude / dB")
                plt.xticks(ticks=np.arange(0, n_frames_total)[::step_width], labels=t_string[::step_width])
                plt.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
                plt.yticks(ticks=np.arange(0, len(cfg.class_labels)), labels=cfg.class_labels)
                plt.title("%s: predicted" % os.path.basename(input_file))
                plt.xlabel("time / s")
                plt.ylabel("instruments")

                # plot reference envelopes
                plt.subplot(2, 1, 2)
                plt.matshow(to_db(envelope_ref), fignum=0, aspect="auto", origin="lower", vmin=-100, vmax=0)
                plt.colorbar(label="amplitude / dB")
                plt.xticks(ticks=np.arange(0, n_frames_total)[::step_width], labels=t_string[::step_width])
                plt.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
                plt.yticks(ticks=np.arange(0, len(cfg.class_labels)), labels=cfg.class_labels)
                plt.title("%s: reference" % os.path.basename(input_file))
                plt.xlabel("time / s")
                plt.ylabel("instruments")

                plt.tight_layout()

                # save plot to file
                plot_file_path = os.path.join(
                    cfg.inference.predictions_dir,
                    os.path.splitext(os.path.basename(input_file))[0] + "_comparison.png",
                )
                fig.savefig(plot_file_path)
                log.info("Saved '%s'." % plot_file_path)

                # compute noise and leakage matrix
                noise_matrix, leakage_matrix = compute_noise_and_leakage_matrices(
                    envelope_pred,
                    envelope_ref,
                    sample_length_per_instrument=5.0,
                    total_sample_length=len(audio) / float(dataset_stats["sr"]),
                )

                # plot matrices
                fmt = ".1f"
                cmap = "YlGnBu"
                x_tick_rotation = 45

                fig, _ = plt.subplots(1, 2, figsize=np.array(list(cfg.figsize)) * 1.5)

                plt.subplot(1, 2, 1)
                sns.heatmap(noise_matrix, annot=True, fmt=fmt, cmap=cmap,
                            xticklabels=list(cfg.class_labels), yticklabels=list(cfg.class_labels))
                plt.xlabel("predictions")
                plt.xticks(rotation=x_tick_rotation)
                plt.ylabel("labels")
                plt.title("noise matrix")
                plt.subplot(1, 2, 2)
                sns.heatmap(leakage_matrix, annot=True, fmt=fmt, cmap=cmap,
                            xticklabels=list(cfg.class_labels), yticklabels=list(cfg.class_labels))
                plt.xlabel("predictions")
                plt.xticks(rotation=x_tick_rotation)
                plt.ylabel("labels")
                plt.title("leakage matrix")

                plt.tight_layout()

                # save plot to file
                plot_file_path = os.path.join(
                    cfg.inference.predictions_dir,
                    os.path.splitext(os.path.basename(input_file))[0] + "_matrices.png",
                )
                fig.savefig(plot_file_path)
                log.info("Saved '%s'." % plot_file_path)


                log.info("Computed noise and leakage matrices.")

            # save predictions to CSV too
            for i_instrument, instrument in enumerate(cfg.class_labels):
                csv_file_path = os.path.join(
                    cfg.inference.predictions_dir,
                    "%s_%s.csv" % (os.path.splitext(os.path.basename(input_file))[0], instrument),
                )
                envelope_pred[i_instrument, :].tofile(csv_file_path, sep=",")
                log.info("Saved '%s'." % csv_file_path)

            # update progress bar
            prog_bar.update(1)

        # close progress bar s.t. it is flushed
        prog_bar.close()

        # finish
        log.info("Finished.")


if __name__ == "__main__":
    inference()
