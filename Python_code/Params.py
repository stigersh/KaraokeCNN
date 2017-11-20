class struct(object):
    pass
options = struct()
options.L = 20  # number of stacked frames
options.H = 60  # Training data resample hop
options.N_BINS = 1025  # number of FFT bins
options.FFT_SIZE = 2 * (options.N_BINS - 1)  # STFT FFT size
options.HOP_SIZE = 512  # STFT hop size
options.N_ITER = 100  # number of iterations