__author__='anna'

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


options = struct()
options.L = 20  # number of stacked frames
options.H = 60  # Training data resample hop
options.N_BINS = 1025  # number of FFT bins
options.FFT_SIZE = 2 * (options.N_BINS - 1)  # STFT FFT size
options.HOP_SIZE = 512  # STFT hop size
options.N_ITER = 100  # number of iterations



# read json, read mix from sample and plot
jsondata = json.load(open('sampleDB.json'))#('medleydb_deepkaraoke.json'))  # dictionary

print(jsondata["mixes"][50]["mix_path"])
# print(data['base_path'])
def get_train_data_per_mix(options,ind,jsondata):
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


    f, t, Zotherxx = signal.stft(other, samplerate, 'hann', options.FFT_SIZE, overlap)
    print('stft')

    # plt.pcolormesh(t, f, np.log(np.abs(Zotherxx) ** 2))
    # plt.title('other STFT Magnitude sqr')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.colorbar()

    f, t, Zvocalsxx = signal.stft(vocals, samplerate, 'hann', options.FFT_SIZE, overlap)
    print('stft')
    # plt.pcolormesh(t, f, np.log(np.abs(Zvocalsxx) ** 2))
    # plt.title('vocals STFT Magnitude sqr')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.colorbar()

    f, t, Zmix_totxx = signal.stft(mix_tot, samplerate, 'hann', options.FFT_SIZE, overlap)

    def sample_to_col_blocks(sig, H, L):
        n_blocks = int(np.floor(sig.shape[1] / H))
        if (n_blocks - 1) * H + L > sig.shape[1]:
            n_blocks -= 1
        return [sig[:, i:i + L] for i in np.arange(0, n_blocks * H, H)]


    voc_blks = sample_to_col_blocks(Zvocalsxx, options.H, options.L)
    other_blks = sample_to_col_blocks(Zotherxx, options.H, options.L)
    mix_blks = sample_to_col_blocks(Zmix_totxx, options.H, options.L)

    bin_mask_blks = [np.abs(voc_blks[i]) > np.abs(other_blks[i]) for i in
                     range(0, len(mix_blks))]  # check that this is an np.array and the output is a vector

    # reshape to row vectors - columns block into single row of serialized columns.
    # reshape concat rows -> thus transpose is called
    def reshape_colsblk_to_row_of_cols(blk):
        return np.reshape(blk.transpose(), [1, blk.size])

    mix_rowvecs = np.squeeze(np.array([reshape_colsblk_to_row_of_cols(blk) for blk in mix_blks]))
    bin_mask_rowvecs = np.squeeze(np.array([reshape_colsblk_to_row_of_cols(blk) for blk in bin_mask_blks]))

    df_mixes = pd.DataFrame(mix_rowvecs)
    df_masks = pd.DataFrame(bin_mask_rowvecs)
    # voc_blks
    # other_blks
    return df_mixes, df_masks


#save to csv

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





