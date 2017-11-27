__author__ = 'anna'

# filemod = "__Inference__"
filemod = "__Train__"

# generate csv files of the training data
# one file for the mix spectograms which is the net input
# one file for the binary masks which are the net output binary labels

# method
# get_train_data_per_mix - mode = 'Train' - generates dataFrames for train data per song
# mode = 'Inference' (otherwise) - generates stft, and stft blocks for inference per song
# inference - generate vocals and other
import json
import os
# Hello numpy
import numpy as np
import soundfile as sf
from scipy import signal
import pandas as pd
import tensorflow as tf
from BaseNet import model


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

    f, t, Smix_tot = signal.spectrogram(mix_tot, samplerate, 'hann', options.FFT_SIZE, overlap)


    #for debug GT: To be removed
    f, t, Sother = signal.spectrogram(other, samplerate, 'hann', options.FFT_SIZE, overlap)
    f, t, Svocals = signal.spectrogram(vocals, samplerate, 'hann', options.FFT_SIZE, overlap)

    if mode == 'Train':
        f, t, Sother = signal.spectrogram(other, samplerate, 'hann', options.FFT_SIZE, overlap)
        f, t, Svocals = signal.spectrogram(vocals, samplerate, 'hann', options.FFT_SIZE, overlap)

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
        #for debugging ground truth inference __________________________________
        voc_blks = sample_to_col_blocks(Svocals, options.inference_H, options.L)
        other_blks = sample_to_col_blocks(Sother, options.inference_H, options.L)
        mix_blks = sample_to_col_blocks(Smix_tot, options.inference_H, options.L)
        bin_mask_blks = [np.abs(voc_blks[i]) > np.abs(other_blks[i]) for i in
                         range(0, len(mix_blks))]

        mix_rowvecs = np.squeeze(np.array([reshape_colsblk_to_row_of_cols(blk) for blk in mix_blks], np.float32))
        bin_mask_rowvecs = np.squeeze(
            np.array([reshape_colsblk_to_row_of_cols(blk) for blk in bin_mask_blks], np.float32))
        # END for debugging ground truth inference __________________________________

        # STFTmix_blks = sample_to_col_blocks(STFTmix_tot, options.inference_H, options.L)
        # mix_stft_rowvecs = np.squeeze(
        #     np.array([reshape_colsblk_to_row_of_cols(blk) for blk in STFTmix_blks], np.float32))

    if mode == 'Train':
        df_mixes = pd.DataFrame(mix_rowvecs, dtype=np.float32)
        df_masks = pd.DataFrame(bin_mask_rowvecs, dtype=np.float32)
        return df_mixes, df_masks
    else:  # inference
        return STFTmix_tot, mix_rowvecs,samplerate,bin_mask_rowvecs


# --------------------------------------------------------------------------------------------------------------
# inference
# process network output into separated signals

#helper methods
def reshape_row_of_cols_to_colsblk(row, col_size):
    assert (row.size % col_size == 0)
    return np.reshape(row, [row.size // col_size, col_size]).transpose()

def voc_prob_to_col_mask(alpha, sigm_prob):
    sigm_prob[sigm_prob>1] =1 #truncate relu
    return (np.sum(sigm_prob, 1) / sigm_prob.shape[1] > alpha).astype(np.float32)


def other_prob_to_col_mask(alpha, sigm_prob):
    sigm_prob[sigm_prob > 1] = 1 #truncate relu
    return (np.sum(sigm_prob, 1) / sigm_prob.shape[1] < 1 - alpha).astype(np.float32)


def rows_masks_to_full_stft_mask(rows, col_size, prob2mask_func, alpha,H,n_stft_cols):#H is the hop
    #prob2mask_func is a method taking a block and alpha and calculating the binary decision for each row

    masks_blks = [reshape_row_of_cols_to_colsblk(row, col_size) for row in rows]
    bin_vecs = np.array([prob2mask_func(alpha,blk) for blk in masks_blks]).transpose()#columns of bin vectors (in float)
    rep_bins = np.repeat(bin_vecs,H,1)
    n_last_cols = n_stft_cols -  bin_vecs.shape[1]*H
    if n_last_cols:
       last_col = np.reshape(bin_vecs[:,-1],[bin_vecs.shape[0],1])
       return np.concatenate((rep_bins, np.repeat(last_col,n_last_cols,1)), axis=1)

    return rep_bins

#final method performing the whole task
def nn_out_to_separated_sigs(mix_stft, masks_rows, alpha, samplerate,options):#check correctness
    overlap = options.FFT_SIZE - options.HOP_SIZE
    # masks_list should be in the size of cols on stft  - this issue should be resolved.
    voc_mask = rows_masks_to_full_stft_mask(masks_rows, options.N_BINS, voc_prob_to_col_mask,options.alpha, options.inference_H, mix_stft.shape[1])
    t, sep_voc = signal.istft(np.multiply(mix_stft, voc_mask), samplerate, 'hann', options.FFT_SIZE, overlap)
    other_mask = rows_masks_to_full_stft_mask(masks_rows, options.N_BINS, other_prob_to_col_mask,options.alpha,options.inference_H,mix_stft.shape[1])
    t, sep_other = signal.istft(np.multiply(mix_stft, other_mask), samplerate, 'hann', options.FFT_SIZE, overlap)

    return sep_voc, sep_other


# --------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    def create_data_csv(jsondata,rng,mixes_filenames,masks_filenmaes):
        f = open(mixes_filenames, "w+")
        f.close()
        f = open(masks_filenmaes, "w+")
        f.close()
        for i in rng:#range(0, len(jsondata["mixes"])):
            df_mixes_i, df_masks_i = process_data_per_mix(options, i, jsondata, 'Train')
            with open(mixes_filenames, 'a') as f:
                df_mixes_i.to_csv(f, header=False, index=False)
            with open(masks_filenmaes, 'a') as f:
                df_masks_i.to_csv(f, header=False, index=False)


    options = opt
    jsondata = json.load(open(options.json_name))
    if filemod == "__Train__":
        # run over all songs and save to csv train validation and test
        # read json, read mix from sample and plot
        tvtsplit = options.train_valid_test_songs_split

        create_data_csv(jsondata, range(0,tvtsplit[0]), options.train_mixes_filename ,
                        options.train_masks_filename )
        create_data_csv(jsondata, range(tvtsplit[0],tvtsplit[0]+tvtsplit[1]),
                        options.validation_mixes_filename, options.validation_masks_filename)
        create_data_csv(jsondata, range(tvtsplit[0]+tvtsplit[1],len(jsondata["mixes"])),
                        options.test_mixes_filename, options.test_masks_filename)

    elif filemod == "__Inference__":
    # --------------------------------------------------------------------------------------------------------------
        x_size = options.L * options.N_BINS
        if not os.path.exists(options.inf_dir):
            os.makedirs(options.inf_dir)

        eval_x = tf.placeholder(tf.float32, name='eval_x', shape=[None, x_size])
        eval_probs,_ = model(eval_x, x_size)
        with tf.Session() as sess:
          new_saver = tf.train.import_meta_graph(options.model_dir+"/model_final.ckpt.meta")
          sess.run(tf.global_variables_initializer())
          # Restore variables from disk.
          new_saver.restore(sess, options.model_dir+"/model_final.ckpt")
          for i in range(0, len(jsondata["mixes"])):
              mixname = os.path.basename(jsondata['base_path'] + jsondata["mixes"][i]["mix_path"])
              STFTmix_tot, mix_rowvecs, samplerate, bin_mask_rowvecs = process_data_per_mix(options, i, jsondata, mode='Inference')

              #inference from GT mask
              sep_vocGT, sep_otherGT = nn_out_to_separated_sigs(STFTmix_tot, bin_mask_rowvecs, options.alpha, samplerate, options)
              sf.write(options.inf_dir+'/'+mixname+'vocGT.wav', sep_vocGT, samplerate)
              sf.write(options.inf_dir+'/'+mixname+'otherGT.wav', sep_otherGT, samplerate)
              #get probs from network function
              probs = sess.run(eval_probs, feed_dict={eval_x: mix_rowvecs})
              #inference from net output
              sep_voc, sep_other = nn_out_to_separated_sigs(STFTmix_tot, probs, options.alpha , samplerate, options)
              sf.write(options.inf_dir+'/'+mixname+'vocAlg.wav', sep_voc, samplerate)
              sf.write(options.inf_dir+'/'+mixname+'otherAlg.wav', sep_other, samplerate)

#     elif filemod == "__InferenceKERAS__":
# # load json and create model
# json_file = open(options.model_dir+'model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(options.model_dir+"/KERAS/weights_final.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='mean_squared_error', optimizer='sgd')
#
# probs = loaded_model.predict_on_batch(mix_rowvecs)