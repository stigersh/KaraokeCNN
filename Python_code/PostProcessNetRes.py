import numpy as np
from scipy import signal
# process network output into separated signals
def voc_prob_to_col_mask(alpha, sigm_prob):
    func = lambda x: 1.0 if (np.sum(x) > alpha) else 0.0
    return np.apply_along_axis(func, 1, sigm_prob)

def other_prob_to_col_mask(alpha, sigm_prob):
    func = lambda x: 1.0 if (np.sum(x) < 1 - alpha) else 0.0
    return np.apply_along_axis(func, 1, sigm_prob)

def nn_out_to_separated_sigs(mix_stft, masks_list, alpha, samplerate, FFT_SIZE, overlap):
    # masks_list should be in the size of cols on stft  - this issue should be resolved.
    voc_mask = np.array([voc_prob_to_col_mask(alpha, sigm_prob) for sigm_prob in masks_list])
    t, sep_voc = signal.istft(mix_stft * voc_mask, samplerate, 'hann', FFT_SIZE, overlap)
    other_mask = np.array([other_prob_to_col_mask(alpha, sigm_prob) for sigm_prob in masks_list])
    t, sep_other = signal.istft(mix_stft * other_mask, samplerate, 'hann', FFT_SIZE, overlap)

    return sep_voc, sep_other

