__author__ = 'anna'
__name__ == "__Train__" #__name__ == "__Inference__"
# generate csv files of the training data
# one file for the mix spectograms which is the net input
# one file for the binary masks which are the net output binary labels

# method
# get_train_data_per_mix - mode = 'Train' - generates dataFrames for train data per song
# mode = 'Inference' (otherwise) - generates stft, and stft blocks for inference per song
# inference - generate vocals and other
import json
# Hello numpy
import numpy as np
import soundfile as sf
from scipy import signal
import pandas as pd
import tensorflow as tf
from BaseNet import eval_net

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class struct(object):
    pass

from Params import options as opt



# function which return
# Train mode: DataFrames of training blocks from a single mix
# Inference mode: and stft full and blks for inference from a single mix
def process_data_per_mix(options, ind, jsondata, mode='Train'):
    print(jsondata["mixes"][ind]["mix_path"])
    mixname = jsondata['base_path'] + jsondata["mixes"][ind]["mix_path"]
    other_stems_str = [jsondata['base_path'] + f for f in jsondata["mixes"][ind]["other_stems"]]
    target_stems_str = [jsondata['base_path'] + f for f in jsondata["mixes"][ind]["target_stems"]]

    # print('pysoundfile time')
    # start = time.time()
    mix, samplerate = sf.read(mixname)
    # end = time.time()
    # print(end - start)

    n_samples = len(mix)

    mono = np.mean(mix, 1)

    # plt.plot(mono)
    # plt.show()

    # create stems mats
    def create_stems_mat(stems_str, n_samples):
        stems_mat = np.zeros((len(stems_str), n_samples))
        for i, f in enumerate(stems_str):
            sig, samplerate = sf.read(f)
            stems_mat[i, :] = np.mean(sig, 1)  # make stereo signal into mono
        return stems_mat

    other_stems_mat = create_stems_mat(other_stems_str, n_samples)
    target_stems_mat = create_stems_mat(target_stems_str, n_samples)

    def mix_stems(stems):
        func = lambda x: x / np.max(np.abs(x))
        stems = np.apply_along_axis(func, 1, stems)  # normalize stem to be max 1 (carefully check the range!! TBD)
        n_stems = stems.shape[0]
        return stems.sum(0) / n_stems  # add all stems with equal weights

    other = mix_stems(other_stems_mat)
    vocals = mix_stems(target_stems_mat)
    mix_tot = 0.5 * other + 0.5 * vocals

    overlap = options.FFT_SIZE - options.HOP_SIZE  # this is in the matlab code, in the paper the overlap is the hop size

    if mode == 'Train':
        f, t, Sother = signal.spectrogram(other, samplerate, 'hann', options.FFT_SIZE, overlap)
        f, t, Svocals = signal.spectrogram(vocals, samplerate, 'hann', options.FFT_SIZE, overlap)
        f, t, Smix_tot = signal.spectrogram(mix_tot, samplerate, 'hann', options.FFT_SIZE, overlap)
        # plt.pcolormesh(t, f, np.log(Smix_tot))
        # plt.title('vocals STFT Magnitude sqr')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.colorbar()

    else:  # inference
        f, t, STFTmix_tot = signal.stft(mix_tot, samplerate, 'hann', options.FFT_SIZE, overlap)


    # ----------------------------------------------------------

    # methods for sampling in reshaping to row vectors
    def sample_to_col_blocks(sig, H, L):
        n_blocks = int(np.floor(sig.shape[1] / H))
        if (n_blocks - 1) * H + L > sig.shape[1]:
            n_blocks -= 1
        return [sig[:, i:i + L] for i in np.arange(0, n_blocks * H, H)]

    # reshape to row vectors - columns block into single row of serialized columns.
    # reshape concat rows -> thus transpose is called

    def reshape_colsblk_to_row_of_cols(blk):
        return np.reshape(blk.transpose(), [1, blk.size])

    # ----------------------------------------------------------

    if mode == 'Train':  # calc masks
        voc_blks = sample_to_col_blocks(Svocals, options.H, options.L)
        other_blks = sample_to_col_blocks(Sother, options.H, options.L)
        mix_blks = sample_to_col_blocks(Smix_tot, options.H, options.L)

        bin_mask_blks = [np.abs(voc_blks[i]) > np.abs(other_blks[i]) for i in
                         range(0, len(mix_blks))]  # check that this is an np.array and the output is a vector
        mix_rowvecs = np.squeeze(np.array([reshape_colsblk_to_row_of_cols(blk) for blk in mix_blks], np.float32))
        bin_mask_rowvecs = np.squeeze(
            np.array([reshape_colsblk_to_row_of_cols(blk) for blk in bin_mask_blks], np.float32))
    else:  # inference
        STFTmix_blks = sample_to_col_blocks(STFTmix_tot, options.inference_H, options.L)
        mix_stft_rowvecs = np.squeeze(
            np.array([reshape_colsblk_to_row_of_cols(blk) for blk in STFTmix_blks], np.float32))

    if mode == 'Train':
        df_mixes = pd.DataFrame(mix_rowvecs, dtype=np.float32)
        df_masks = pd.DataFrame(bin_mask_rowvecs, dtype=np.float32)
        return df_mixes, df_masks
    else:  # inference
        return STFTmix_tot, mix_stft_rowvecs,samplerate


# --------------------------------------------------------------------------------------------------------------
# inference

def reshape_row_of_cols_to_colsblk(row, col_size):
    assert (row.size % col_size == 0)
    return np.reshape(row, [row.size // col_size, col_size]).transpose()


# process network output into separated signals
def voc_prob_to_col_mask(alpha, sigm_prob):
    return (np.sum(sigm_prob, 1) / sigm_prob.shape(1) > alpha).astype(np.float32)


def other_prob_to_col_mask(alpha, sigm_prob):
    return (np.sum(sigm_prob, 1) / sigm_prob.shape(1) < 1 - alpha).astype(np.float32)


def rows_masks_to_full_stft_mask(rows, col_size, prob2mask_func,H,n_stft_cols):#H is the hop
    masks_blks = [reshape_row_of_cols_to_colsblk(row, col_size) for row in rows]
    bin_vecs = np.array([prob2mask_func(blk) for blk in masks_blks]).transpose()#columns of bin vectors (in float)
    rep_bins = np.repeat(bin_vecs,H,1)
    n_last_cols = n_stft_cols -  bin_vecs.shape[1]*H
    if n_last_cols:
       last_col =   np.reshape(bin_vecs[:,-1],[bin_vecs.shape[0],1])
       return np.concatenate((rep_bins, np.repeat(last_col,n_last_cols,1)), axis=1)

    return rep_bins


def nn_out_to_separated_sigs(mix_stft, masks_rows, alpha, samplerate,options):#check correctness
    overlap = options.FFT_SIZE - options.HOP_SIZE
    # masks_list should be in the size of cols on stft  - this issue should be resolved.
    voc_mask = rows_masks_to_full_stft_mask(masks_rows, options.L, voc_prob_to_col_mask,options.inference_H,mix_stft.shape[1])
    t, sep_voc = signal.istft(np.multiply(mix_stft, voc_mask), samplerate, 'hann', options.FFT_SIZE, overlap)
    other_mask = rows_masks_to_full_stft_mask(masks_rows, options.L, other_prob_to_col_mask,options.inference_H,mix_stft.shape[1])
    t, sep_other = signal.istft(np.multiply(mix_stft, other_mask), samplerate, 'hann', options.FFT_SIZE, overlap)

    return sep_voc, sep_other


# --------------------------------------------------------------------------------------------------------------
options = opt
jsondata = json.load(open('sampleDB.json'))  # ('medleydb_deepkaraoke.json'))  # dictionary
if __name__ == "__Train__":
    # run over all training songs and save to csv
    # read json, read mix from sample and plot
    train_mixes_filename = 'train_mixes.csv'
    train_masks_filename = 'train_masks.csv'

    for i in range(0, len(jsondata["mixes"])):
        df_mixes_i, df_masks_i = process_data_per_mix(options, i, jsondata, 'Train')
        with open(train_mixes_filename, 'a') as f:
            df_mixes_i.to_csv(f, header=False, index=False)
        with open(train_masks_filename, 'a') as f:
            df_masks_i.to_csv(f, header=False, index=False)


# datafilename = 'train_data_sample.pckl'
# f = open(datafilename, 'wb')
# pickle.dump(train_data, f)
# f.close()
else: #inference
# --------------------------------------------------------------------------------------------------------------
    saver = tf.train.Saver()
    with tf.Session() as sess:
      # Restore variables from disk.
      last_iter = 20 #change that
      saver.restore(sess, "/save_model/model_iter_"+str(last_iter)+".ckpt")
      STFTmix_tot, mix_stft_rowvecs,samplerate = process_data_per_mix(options, 0, jsondata, mode='Inference')
      probs_eval = eval_net(mix_stft_rowvecs, sess)
      nn_out_to_separated_sigs(STFTmix_tot, probs_eval, options.alpha , samplerate, options)