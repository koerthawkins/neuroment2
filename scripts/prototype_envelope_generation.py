""" Compare different envelope types with a reference RMS envelope computed in time domain.

    Basically we compare 4 different types of signal envelopes:
    1. RMS, computed directly in time domain
    2. STFT, normalized, squared, summed
    3. CQT, squared, divided by total number of bins
    4. CQT, normalized by the respective filter lengths of each bins, squared, summed up
"""

import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def butter_coefficients(cutoff, fs, filter_type, order=5):
    """ Creates coefficients for a Butterworth filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return b, a


def butter_filter(data, cutoff, fs, filter_type, order=5):
    """ Filters audio with a 5th-order Butterworth filter.
    """
    b, a = butter_coefficients(cutoff, fs, filter_type, order=order)
    y = lfilter(b, a, data)
    return y


def read_audio(file_path, sampling_rate, audio_length_in_s):
    """ Reads audio file from disk and checks the sampling rate.
    """
    file_sr = lb.get_samplerate(file_path)

    if file_sr < sampling_rate:
        print(
            "Sample rate of file %s is only %d, but we use sample rate %d!"
            % (file_path, file_sr, sampling_rate)
        )

    audio, _ = lb.load(
        file_path,
        sr=sampling_rate,
        mono=True,
    )

    # cut off all samples after audio_length_in_s
    if len(audio) > int(sampling_rate * audio_length_in_s):
        audio = audio[:int(sampling_rate * audio_length_in_s)]

    return audio


##### PARAMETERS #######################
FRAME_SIZE = 2048
HOP_SIZE = FRAME_SIZE // 4
N_BINS_PER_OCTAVE = [12, 24, 48]  # bass, mid, high
N_OCTAVES = [3, 2, 3]  # bass, mid, high
F_MIN = 27.5
AUDIO_LENGTH_IN_S = 5.0
SAMPLE_RATE = 44100
WINDOW = "hann"  # recommended options: "hann", "boxcar" (= rectangular)
N_MELS = 128

TEST_DATA = [
    # fields: name_of_audio, audio
    [
        "white_noise_uniformly_sampled",
        np.random.uniform(-1.0, 1.0, (int(SAMPLE_RATE * AUDIO_LENGTH_IN_S),)),
    ],
    [
        "exponential_sweep_40hz-4kHz",
        lb.chirp(
            40.0,
            4000.0,
            sr=SAMPLE_RATE,
            length=SAMPLE_RATE * AUDIO_LENGTH_IN_S,
            linear=False,
        ),
    ],
    # [
        # "beatles_let-it-be",
        # read_audio("data/input/LetItBe_20210102.wav", 44100, AUDIO_LENGTH_IN_S)
    # ],
    [
        "trumpet",
        read_audio(lb.util.ex("trumpet"), 44100, AUDIO_LENGTH_IN_S)
    ]
]
###########################################

# compute total number of octaves
n_total_octaves = int(np.sum(N_OCTAVES))

def compute_cqts(audio):
    """ Compute 3 different CQTs: 1 for bass frequencies, 1 for mids, 1 for highs.
    """
    cqt_b = lb.cqt(
        audio,
        hop_length=HOP_SIZE,
        n_bins=N_BINS_PER_OCTAVE[0] * n_total_octaves,
        bins_per_octave=N_BINS_PER_OCTAVE[0],
        fmin=F_MIN,
        window=WINDOW,
        scale=False,
    )
    cqt_b = np.abs(cqt_b)

    cqt_m = lb.cqt(
        audio,
        hop_length=HOP_SIZE,
        n_bins=N_BINS_PER_OCTAVE[1] * n_total_octaves,
        bins_per_octave=N_BINS_PER_OCTAVE[1],
        fmin=F_MIN,
        window=WINDOW,
        scale=False,
    )
    cqt_m = np.abs(cqt_m)

    cqt_h = lb.cqt(
        audio,
        hop_length=HOP_SIZE,
        n_bins=N_BINS_PER_OCTAVE[2] * n_total_octaves,
        bins_per_octave=N_BINS_PER_OCTAVE[2],
        fmin=F_MIN,
        window=WINDOW,
        scale=False,
    )
    cqt_h = np.abs(cqt_h)

    return cqt_b, cqt_m, cqt_h


# compute filter lengths for each bin in the CQTs
cqt_lengths_b = lb.filters.constant_q_lengths(sr=SAMPLE_RATE,
                                              n_bins=N_BINS_PER_OCTAVE[0] * n_total_octaves,
                                              bins_per_octave=N_BINS_PER_OCTAVE[0],
                                              fmin=F_MIN,
                                              window=WINDOW, )
cqt_lengths_m = lb.filters.constant_q_lengths(sr=SAMPLE_RATE,
                                              n_bins=N_BINS_PER_OCTAVE[1] * n_total_octaves,
                                              bins_per_octave=N_BINS_PER_OCTAVE[1],
                                              fmin=F_MIN,
                                              window=WINDOW, )
cqt_lengths_h = lb.filters.constant_q_lengths(sr=SAMPLE_RATE,
                                              n_bins=N_BINS_PER_OCTAVE[2] * n_total_octaves,
                                              bins_per_octave=N_BINS_PER_OCTAVE[2],
                                              fmin=F_MIN,
                                              window=WINDOW, )

# compute center frequencies of each CQT bin
cqt_freqs_b = lb.cqt_frequencies(n_bins=N_BINS_PER_OCTAVE[0] * n_total_octaves,
                                 fmin=F_MIN,
                                 bins_per_octave=N_BINS_PER_OCTAVE[0])
cqt_freqs_m = lb.cqt_frequencies(n_bins=N_BINS_PER_OCTAVE[1] * n_total_octaves,
                                 fmin=F_MIN,
                                 bins_per_octave=N_BINS_PER_OCTAVE[1])
cqt_freqs_h = lb.cqt_frequencies(n_bins=N_BINS_PER_OCTAVE[2] * n_total_octaves,
                                 fmin=F_MIN,
                                 bins_per_octave=N_BINS_PER_OCTAVE[2])

# compute CQT bin indices for stacked variant of CQT
bins_b = np.arange(N_BINS_PER_OCTAVE[0] * 0, N_BINS_PER_OCTAVE[0] * 3)
bins_m = np.arange(N_BINS_PER_OCTAVE[1] * 3, N_BINS_PER_OCTAVE[1] * 5)
bins_h = np.arange(N_BINS_PER_OCTAVE[2] * 5, N_BINS_PER_OCTAVE[2] * 8)

# keep only CQT lengths of corresponding bins
cqt_lengths_b = cqt_lengths_b[bins_b]
cqt_lengths_m = cqt_lengths_m[bins_m]
cqt_lengths_h = cqt_lengths_h[bins_h]

# keep only CQT center frequencies of corresponding bins
cqt_freqs_b = cqt_freqs_b[bins_b]
cqt_freqs_m = cqt_freqs_m[bins_m]
cqt_freqs_h = cqt_freqs_h[bins_h]

# stack bass, mid and high CQT filter lengths/center frequencies to vectors
cqt_lengths_stacked = np.concatenate((cqt_lengths_b, cqt_lengths_m, cqt_lengths_h), axis=0)
cqt_freqs_stacked = np.concatenate((cqt_freqs_b, cqt_freqs_m, cqt_freqs_h), axis=0)

# if CQT frequency vector is not steadily increasing we did do something wrong
assert np.all(np.diff(cqt_freqs_stacked) > 0.0)

# analyze envelopes for each audio data
for (audio_name, audio) in TEST_DATA:
    print("Analyzing audio %s ..." % audio_name)

    # filter audio below minimum CQT frequency
    audio = butter_filter(audio, cqt_freqs_stacked[0], SAMPLE_RATE, "high")

    # filter audio above maximum CQT frequency
    audio = butter_filter(audio, cqt_freqs_stacked[-1], SAMPLE_RATE, "low")

    cqt_b, cqt_m, cqt_h = compute_cqts(audio)

    # keep only CQT values of corresponding bins
    cqt_b = cqt_b[bins_b, :]
    cqt_m = cqt_m[bins_m, :]
    cqt_h = cqt_h[bins_h, :]

    # stack the CQT bins
    cqt_stacked = np.concatenate((cqt_b, cqt_m, cqt_h), axis=0)

    # envelope 1: RMS envelope using time domain. no windowing here!
    rms_envelope = lb.feature.rms(audio, frame_length=FRAME_SIZE, hop_length=HOP_SIZE, center=True)[0, :] ** 2.0

    # envelope 2: STFT, squared and sum, divided by total number of bins
    stft = np.abs(lb.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, window=WINDOW, center=True,))
    stft /= FRAME_SIZE
    stft_envelope = np.sum(stft ** 2.0, axis=0)

    # envelope 3: mel-filtered STFT, squared and sum
    stft_mel = np.abs(lb.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, window=WINDOW, center=True,))
    stft_mel /= FRAME_SIZE
    mel_filter_bank = lb.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=FRAME_SIZE,
        n_mels=N_MELS,
        fmin=cqt_freqs_stacked[0],
        fmax=cqt_freqs_stacked[-1],
        norm=1.0,
    )
    mel = np.dot(mel_filter_bank, stft_mel)
    mel_envelope = np.sum(mel ** 2.0, axis=0)

    # envelope 4: CQT, each bin divided by its own filter length, squared, summed
    cqt_envelope = np.sum(((cqt_stacked / cqt_lengths_stacked[:, None]) ** 2.0), axis=0)

    # plot results
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 9))

    # plot stacked CQT
    plt.subplot(1, 2, 1)
    plt.imshow(cqt_stacked, origin="lower", aspect="auto")
    plt.xlabel("frames")
    plt.ylabel("CQT bins")
    plt.colorbar(label="CQT amplitude")
    plt.title("Stacked CQT")

    # plot envelopes
    plt.subplot(4, 2, 2)
    plt.plot(rms_envelope)
    plt.grid()
    plt.title("RMS envelope")

    plt.subplot(4, 2, 4)
    plt.plot(stft_envelope)
    plt.grid()
    plt.title("STFT envelope")

    plt.subplot(4, 2, 6)
    plt.plot(mel_envelope)
    plt.grid()
    plt.xlabel("frames")
    plt.title("Mel envelope")

    plt.subplot(4, 2, 8)
    plt.plot(cqt_envelope)
    plt.grid()
    plt.xlabel("frames")
    plt.title("CQT envelope")

    # save plot and show it
    plt.suptitle(audio_name)
    plt.tight_layout()
    plt.savefig("%s.png" % audio_name)
    plt.show()

    # compute the ratio of the average envelope values to the reference RMS envelope
    stft_to_rms = np.mean(stft_envelope) / np.mean(rms_envelope)
    mel_to_rms = np.mean(mel_envelope) / np.mean(rms_envelope)
    cqt_to_rms = np.mean(cqt_envelope) / np.mean(rms_envelope)

    # print the ratio
    print("Ratio between average STFT envelope and average RMS envelope: %.4f" % stft_to_rms)
    print("Ratio between average Mel envelope type 2 and average RMS envelope: %.4f" % mel_to_rms)
    print("Ratio between average CQT envelope type 2 and average RMS envelope: %.4f" % cqt_to_rms)
    print("Finished analysis of file.")
