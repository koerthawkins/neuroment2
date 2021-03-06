import librosa as lb
import numpy as np
from omegaconf import DictConfig


def compute_reference_envelopes(
    cfg: DictConfig,
    audio: np.ndarray,
    feature_generator_cfg: dict,
    n_frames_total: int,
    sample_length_per_instrument: float,
    total_sample_length: float,
):
    """Computes the reference envelopes for Sequence_1.flac and returns them.

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
    # doing it this way we know that the buffering works exactly the same way like for the predictions
    rms = lb.feature.rms(
        audio,
        frame_length=feature_generator_cfg["Mix"]["dft_size"],
        hop_length=feature_generator_cfg["Mix"]["hopsize"],
        center=feature_generator_cfg["center"],
    )[0, :]

    # assign RMS values to instruments
    # compute start indices of instruments in sequences
    sample_start_indices = (
        np.arange(0, n_instruments)
        * (sample_length_per_instrument / total_sample_length)
        * n_frames_total
    )
    sample_start_indices = list(sample_start_indices.astype(int))

    # we add the last frame index to sample_start_indices because it's easier for the loop later
    sample_start_indices += [n_frames_total]

    # go through start indices
    for i_instrument in range(n_instruments):
        current_frame_indices = np.arange(
            sample_start_indices[i_instrument], sample_start_indices[i_instrument + 1]
        )
        envelopes[i_instrument, current_frame_indices] = rms[current_frame_indices]

    return envelopes


def compute_noise_and_leakage_matrices(
    predicted_envelopes: np.ndarray,
    label_envelopes: np.ndarray,
    sample_length_per_instrument: float,
    total_sample_length: float,
    level_range_in_db: tuple = (
        -100.0,
        0.0,
    ),
):
    """Computes the noise and leakage matrices for predicted and reference
       envelopes of a sequence sample.

        For NOISE we compare each the predicted envelope with the label envelope. The label
        envelope can either be from the single instrument playing, or 0 (instrument doesn't play).
        For LEAKAGE we check how much from the instrument playing leaks into the instruments
        currently not playing.

        We need to limit the dynamic range here, otherwise there can be too many outliers
        for the results to have meaning.

    Args:
        predicted_envelopes : np.ndarray
            Predicted envelopes of shape [n_instruments, n_time_frames].
            Expected to be clipped to level_range_in_db.
        label_envelopes: : np.ndarray
            Label envelopes of shape [n_instruments, n_time_frames]
            Expected to be clipped to level_range_in_db.
        sample_length_per_instrument : float
            How long each instrument is played in the analyzed sample in seconds.
            We assume that all instruments are played equally long!
        total_sample_length : float
            The total length of the sample in seconds.
        level_range_in_db : tuple/list
            The level range in dB we want to limit the noise and leakage matrices too.
            The 1st element is the min value in dB, the 2nd element the max value in dB.

    Returns:
        noise_matrix : np.ndarray
            Noise matrix of shape [n_instruments, n_instruments].
            Labels are on axis 0 (rows), predictions are on axis 1 (columns).
        leakage_matrix : np.ndarray
            Leakage matrix of shape [n_instruments, n_instruments].
            Labels are on axis 0 (rows), predictions are on axis 1 (columns).
    """
    # determine the indices in envelope matrices where instruments start playing
    n_instruments, n_frames = predicted_envelopes.shape
    sample_start_indices = (
        np.arange(0, n_instruments)
        * (sample_length_per_instrument / total_sample_length)
        * n_frames
    )
    sample_start_indices = list(sample_start_indices.astype(int))

    # we add the last frame index to sample_start_indices because it's easier for the loop later
    sample_start_indices += [n_frames]

    # init noise and leakage matrices
    # labels are on axis 0, predictions on axis 1
    noise_matrix = np.zeros(shape=[n_instruments, n_instruments])
    leakage_matrix = np.zeros(shape=[n_instruments, n_instruments])

    for i_labeled_instrument in range(n_instruments):
        # get frame indices of current instrument
        frame_indices = np.arange(
            sample_start_indices[i_labeled_instrument],
            sample_start_indices[i_labeled_instrument + 1],
        )

        for i_predicted_instrument in range(n_instruments):
            # get labeled and predicted envelope of current time segment
            cur_label_env_noise = label_envelopes[i_predicted_instrument, frame_indices]
            cur_label_env_leakage = label_envelopes[i_labeled_instrument, frame_indices]
            cur_pred_env = predicted_envelopes[i_predicted_instrument, frame_indices]

            # convert to db
            cur_label_env_noise = to_db(
                cur_label_env_noise,
                min_db=level_range_in_db[0],
                max_db=level_range_in_db[1],
            )
            cur_label_env_leakage = to_db(
                cur_label_env_leakage,
                min_db=level_range_in_db[0],
                max_db=level_range_in_db[1],
            )
            cur_pred_env = to_db(
                cur_pred_env, min_db=level_range_in_db[0], max_db=level_range_in_db[1]
            )

            # compute the value in the noise matrix by averaging the difference
            # between predicted and labelled envelope over time
            noise_diff_over_time_in_db = cur_pred_env - cur_label_env_noise

            if i_labeled_instrument != i_predicted_instrument:
                # we have an element off the main diagonal
                # we shift the range of the computed difference into level_range_in_db
                noise_diff_over_time_in_db += level_range_in_db[0]
            else:
                # when predicted instrument matches label instrument we must FLIP the value range
                # this way the colour code (light -> good, dark -> bad) is equal for values on and off
                # the main diagonal
                noise_diff_over_time_in_db = (
                    level_range_in_db[0]
                    - noise_diff_over_time_in_db
                    - level_range_in_db[1]
                )

            # add noise value to noise matrix
            noise_matrix[i_labeled_instrument, i_predicted_instrument] = float(
                np.mean(noise_diff_over_time_in_db)
            )

            # compute leakage
            leakage_diff_over_time_in_db = cur_pred_env - cur_label_env_leakage

            # leakage only works for frames which have an energy above the min level in db
            # so if energy of prediction is not above min level in db we have no leakage
            leakage_diff_over_time_in_db[
                cur_pred_env < (level_range_in_db[0] + 0.1)
            ] = level_range_in_db[0]

            # average the leakage difference over time to get leakage matrix value
            leakage_matrix[i_labeled_instrument, i_predicted_instrument] = np.mean(
                leakage_diff_over_time_in_db
            )

    # main diagonal elements of leakage don't make sense, so we set them to nan
    leakage_matrix[np.diag_indices_from(leakage_matrix)] = np.nan

    return noise_matrix, leakage_matrix


def to_db(spectrum: np.ndarray, min_db: float = -100.0, max_db: float = 0.0):
    """Converts a spectrum from linear to dB and returns it."""
    spectrum_log = 20.0 * np.log10(spectrum + 1e-12)
    if min_db is not None:
        return np.clip(spectrum_log, min_db, max_db)
    else:
        return 20.0 * np.log10(spectrum + 1e-12)


def to_linear(spectrum: np.ndarray):
    """Converts a spectrum from dB to linear and returns it."""
    return np.power(10.0, spectrum * 0.05)
