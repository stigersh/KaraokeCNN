__author__='anna'

#generate csv files of the training data
#one file for the mix spectograms which is the net input
#one file for the binary masks which are the net output binary labels



# write and read data
import pickle
import json

# Hello numpy
import numpy as np
import soundfile as sf
from scipy import signal
import pandas as pd

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class struct(object):
    pass

from Params import options as opt

options = opt


# read json, read mix from sample and plot
jsondata = json.load(open('sampleDB.json'))#('medleydb_deepkaraoke.json'))  # dictionary

# main function which return DataFrame of training blocks from a single song
def get_train_data_per_mix(options,ind,jsondata):
    print(jsondata["mixes"][ind]["mix_path"])
    mixname = jsondata['base_path'] + jsondata["mixes"][ind]["mix_path"]
    other_stems_str = [jsondata['base_path'] + f for f in jsondata["mixes"][ind]["other_stems"]]
    target_stems_str = [jsondata['base_path'] + f for f in jsondata["mixes"][ind]["target_stems"]]

    #print('pysoundfile time')
    #start = time.time()
    mix, samplerate = sf.read(mixname)
    #end = time.time()
    #print(end - start)

    n_samples = len(mix)

    mono = np.mean(mix,1)

    #plt.plot(mono)
    #plt.show()


    # create stems mats
    def create_stems_mat(stems_str, n_samples):
        stems_mat = np.zeros((len(stems_str), n_samples))
        for i, f in enumerate(stems_str):
            sig, samplerate = sf.read(f)
            stems_mat[i, :] = np.mean(sig,1)  # make stereo signal into mono
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


    f, t, Sother = signal.spectrogram(other, samplerate, 'hann', options.FFT_SIZE, overlap)
    # if isinstance(Sother, complex):
    #     print('fuck complex')

    f, t, Svocals = signal.spectrogram(vocals, samplerate, 'hann', options.FFT_SIZE, overlap)

    # plt.pcolormesh(t, f, np.log(Svocals))
    # plt.title('vocals STFT Magnitude sqr')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.colorbar()

    f, t, Smix_tot = signal.spectrogram(mix_tot, samplerate, 'hann', options.FFT_SIZE, overlap)

    def sample_to_col_blocks(sig, H, L):
        n_blocks = int(np.floor(sig.shape[1] / H))
        if (n_blocks - 1) * H + L > sig.shape[1]:
            n_blocks -= 1
        return [sig[:, i:i + L] for i in np.arange(0, n_blocks * H, H)]


    voc_blks = sample_to_col_blocks(Svocals, options.H, options.L)
    print (voc_blks[0].shape)
    other_blks = sample_to_col_blocks(Sother, options.H, options.L)
    mix_blks = sample_to_col_blocks(Smix_tot, options.H, options.L)

    bin_mask_blks = [np.abs(voc_blks[i]) > np.abs(other_blks[i]) for i in
                     range(0, len(mix_blks))]  # check that this is an np.array and the output is a vector

    # reshape to row vectors - columns block into single row of serialized columns.
    # reshape concat rows -> thus transpose is called
    def reshape_colsblk_to_row_of_cols(blk):
        return np.reshape(blk.transpose(), [1, blk.size])

    mix_rowvecs = np.squeeze(np.array([reshape_colsblk_to_row_of_cols(blk) for blk in mix_blks],np.float32))
    bin_mask_rowvecs = np.squeeze(np.array([reshape_colsblk_to_row_of_cols(blk) for blk in bin_mask_blks],np.float32))

    # if isinstance(mix_rowvecs[0,0], complex):
    #     print('fuck complex')

    df_mixes = pd.DataFrame(mix_rowvecs, dtype= np.float32)
    # print(df_mixes.dtypes)
    df_masks = pd.DataFrame(bin_mask_rowvecs, dtype= np.float32)
    # voc_blks
    # other_blks
    return df_mixes, df_masks


# run over all training songs and save to csv

train_mixes_filename = 'train_mixes.csv'
train_masks_filename = 'train_masks.csv'


for i in range(0,len(jsondata["mixes"])):
    df_mixes_i, df_masks_i = get_train_data_per_mix(options,i,jsondata)
    with open(train_mixes_filename, 'a') as f:
        df_mixes_i.to_csv(f, header=False, index=False)
    with open(train_masks_filename, 'a') as f:
        df_masks_i.to_csv(f, header=False, index=False)


# datafilename = 'train_data_sample.pckl'
# f = open(datafilename, 'wb')
# pickle.dump(train_data, f)
# f.close()





