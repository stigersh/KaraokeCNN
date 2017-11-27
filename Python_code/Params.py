class struct(object):
    pass
options = struct()
options.L = 20  # number of stacked frames
options.H = 60  # Training data resample hop
options.inference_H = 19 #inference resample hop of stft
options.N_BINS = 1025  # number of FFT bins
options.FFT_SIZE = 2 * (options.N_BINS - 1)  # STFT FFT size
options.HOP_SIZE = 512  # STFT hop size
options.N_ITER = 100  # number of iterations
options.alpha = 0.5

options.json_name =  'sampleDB.json' # 'medleydb_deepkaraoke.json'
options.train_valid_test_songs_split = [45, 5, 13]
options.train_mixes_filename = 'train_mixes.csv' #'my_csv.csv'
options.train_masks_filename = 'train_masks.csv' #'my_csv_labels.csv'
options.valid_mixes_filename = 'valid_mixes.csv'
options.valid_masks_filename = 'valid_masks.csv'
options.test_mixes_filename  = 'test_mixes.csv'
options.test_masks_filename  = 'test_masks.csv'



#net options
options.n_epochs = 1
options.lr = 0.5
options.reg = 1e-5
options.batch_size = 100
options.model_dir = 'save_model'
options.save_iters = 1000

options.inference_dir = 'inf_dir'