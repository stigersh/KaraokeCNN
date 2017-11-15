__author__='anna'


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class struct(object):
    pass


options = struct();
options.L = 20;  # number of stacked frames
options.H = 60;  # Training data resample hop
options.N_BINS = 1025;  # number of FFT bins
options.FFT_SIZE = 2 * (options.N_BINS - 1);  # STFT FFT size
options.HOP_SIZE = 512;  # STFT hop size
options.N_ITER = 100;  # number of iterations


# Hello numpy
import numpy as np
from scipy import signal

import json
from pprint import pprint
import time
import matplotlib.pyplot as plt
import soundfile as sf

# read json, read mix from sample and plot
jsondata = json.load(open('medleydb_deepkaraoke.json'))  # dictionary

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

    train_data = struct()
    train_data.mix_blks = mix_blks
    train_data.bin_mask_blks = bin_mask_blks
    train_data.voc_blks = voc_blks
    train_data.other_blks = other_blks
    return train_data

# write and read data
import pickle


train_data = get_train_data_per_mix(options,50,jsondata)

f = open('train_data_pycharm.pckl', 'wb')
pickle.dump(train_data, f)
f.close()

f = open('train_data_pycharm.pckl', 'rb')
train_data_in = pickle.load(f)
f.close()



