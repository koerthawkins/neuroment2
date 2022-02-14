import librosa as lb
import numpy as np
from omegaconf import DictConfig


def compute_reference_envelopes(cfg: DictConfig, audio: np.ndarray, feature_generator_cfg: dict, n_frames_total: int, sample_length_per_instrument: float, total_sample_length: float):
    """ Computes the reference envelopes for Sequence_1.flac and returns them.

        We assume that each instrument plays exactly for 5s.
    """
    # init envelopes
    n_instruments = len(cfg.class_labels)
    envelopes = np.zeros(shape=[n_instruments, n_frames_total])

    # for i_instrument, _ in enumerate(cfg.class_labels):
    #     start_sample = 5.0 * feature_generator_cfg["Mix"]["sr"] * i_instrument
    #     stop_sample = np.min([5.0 * feature_generator_cfg["Mix"]["sr"] * (i_instrument + 1), len(audio)])
    #
    #     # TODO this is not perfectly accurate! it doesn't consider the initial frame AND centering!
    #     start_frame = int(start_sample // feature_generator_cfg["Mix"]["hopsize"])
    #
    #     audio_current_instrument = audio[int(start_sample):int(stop_sample)]
    #
    #     rms = lb.feature.rms(
    #         audio_current_instrument,
    #         frame_length=feature_generator_cfg["Mix"]["dft_size"],
    #         hop_length=feature_generator_cfg["Mix"]["hopsize"],
    #         center=feature_generator_cfg["center"],
    #     )[0, :]
    #
    #     envelopes[i_instrument, start_frame:start_frame + len(rms)] = rms

    # compute RMS for WHOLE sample
    # this way we know that the buffering works exactly the same way like for the predictions
    rms = lb.feature.rms(
        audio,
        frame_length=feature_generator_cfg["Mix"]["dft_size"],
        hop_length=feature_generator_cfg["Mix"]["hopsize"],
        center=feature_generator_cfg["center"],
    )[0, :]

    # assign RMS values to instruments
    # compute start indices of instruments in sequences
    sample_start_indices = np.arange(0, n_instruments) * (sample_length_per_instrument / total_sample_length) * n_frames_total
    sample_start_indices = list(sample_start_indices.astype(int))

    # we add the last frame index to sample_start_indices because it's easier for the loop later
    sample_start_indices += [n_frames_total]

    # go through start indices
    for i_instrument in range(n_instruments):
        current_frame_indices = np.arange(sample_start_indices[i_instrument],
                                          sample_start_indices[i_instrument + 1])
        envelopes[i_instrument, current_frame_indices] = rms[current_frame_indices]

    return envelopes


def compute_noise_matrix(predicted_envelopes: np.ndarray, label_envelopes: np.ndarray, sample_length_per_instrument: float, total_sample_length: float):
    """ Computes the noise matrix for predicted and reference envelopes of a sequence sample.

        Noise is termed as predictions where there should be silence (in labels).

    Args:
        cfg:
        predicted_envelopes:
        label_envelopes:
        sample_length:

    Returns:

    """
    # compute start indices of instruments in sequences
    n_instruments, n_frames = predicted_envelopes.shape
    sample_start_indices = np.arange(0, n_instruments) * (sample_length_per_instrument / total_sample_length) * n_frames
    sample_start_indices = list(sample_start_indices.astype(int))

    # we add the last frame index to sample_start_indices because it's easier for the loop later
    sample_start_indices += [n_frames]

    # init confusion matrix
    # labels are on axis 0, predictions on axis 1
    noise_matrix = np.zeros(shape=[n_instruments, n_instruments])

    for i_labeled_instrument in range(n_instruments):
        # get current frame indices
        frame_indices = np.arange(sample_start_indices[i_labeled_instrument],
                                  sample_start_indices[i_labeled_instrument + 1])

        for i_predicted_instrument in range(n_instruments):
            # get labeled and predicted envelope of current time segment
            # for noise we compare each ...
            cur_label_env = label_envelopes[i_predicted_instrument, frame_indices]
            cur_pred_env = predicted_envelopes[i_predicted_instrument, frame_indices]

            # compute difference in envelopes over time in dB
            difference_over_time_in_db = np.abs(to_db(cur_label_env) - to_db(cur_pred_env))
            average_difference_in_db = np.mean(difference_over_time_in_db)

            # difference_over_time_in_db = np.abs(cur_label_env - cur_pred_env)
            # average_difference_in_db = np.mean(to_db(difference_over_time_in_db))

            # add the entry to the confusion matrix
            noise_matrix[i_labeled_instrument, i_predicted_instrument] = float(average_difference_in_db)

    return noise_matrix


def compute_leakage_matrix(predicted_envelopes: np.ndarray, label_envelopes: np.ndarray, sample_length_per_instrument: float, total_sample_length: float):
    """ Computes the leakage matrix for predicted and reference envelopes of a sequence sample.

        Leakage is termed as predictions in instruments that are NOT the currently playing instrument.

    Args:
        cfg:
        predicted_envelopes:
        label_envelopes:
        sample_length:

    Returns:

    """
    # compute start indices of instruments in sequences
    n_instruments, n_frames = predicted_envelopes.shape
    sample_start_indices = np.arange(0, n_instruments) * (sample_length_per_instrument / total_sample_length) * n_frames
    sample_start_indices = list(sample_start_indices.astype(int))

    # we add the last frame index to sample_start_indices because it's easier for the loop later
    sample_start_indices += [n_frames]

    # init confusion matrix
    # labels are on axis 0, predictions on axis 1
    leakage_matrix = np.zeros(shape=[n_instruments, n_instruments])

    for i_labeled_instrument in range(n_instruments):
        # get current frame indices
        frame_indices = np.arange(sample_start_indices[i_labeled_instrument],
                                  sample_start_indices[i_labeled_instrument + 1])

        for i_predicted_instrument in range(n_instruments):
            # get labeled and predicted envelope of current time segment
            # for leakage we always use the same label (for a current sample) and go through the
            # different predictions of each instrument
            cur_label_env = label_envelopes[i_labeled_instrument, frame_indices]
            cur_pred_env = predicted_envelopes[i_predicted_instrument, frame_indices]

            # compute difference in envelopes over time in dB
            difference_over_time_in_db = np.abs(to_db(cur_label_env) - to_db(cur_pred_env))
            average_difference_in_db = np.mean(difference_over_time_in_db)

            # difference_over_time_in_db = np.abs(cur_label_env - cur_pred_env)
            # average_difference_in_db = np.mean(to_db(difference_over_time_in_db))

            # add the entry to the confusion matrix
            leakage_matrix[i_labeled_instrument, i_predicted_instrument] = float(average_difference_in_db)

    return leakage_matrix


def to_db(spectrum: np.ndarray):
    """ Converts a spectrum from linear to dB and returns it.
    """
    return 20.0 * np.log10(spectrum + 1e-12)
