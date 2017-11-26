from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers
from keras.utils import np_utils
from keras.models import model_from_json

import numpy as np
import pickle
from Params import options as opt
import os
np.random.seed(123)  # for reproducibility

from DataClass import DataClass

def build_model(x_size,reg):

    model = Sequential()
    model.add(Dense(x_size,input_dim=64,
                 activation='relu',kernel_regularizer=regularizers.l2(reg)))

    # model.add(Dense(x_size,
    #              activation='relu',kernel_regularizer=regularizers.l2(reg)))
    # model.add(Dense(x_size,
    #              activation='relu',kernel_regularizer=regularizers.l2(reg)))

    return model

#----------------------------------
if __name__ == "__main__":
    options = opt

    n_iters = options.n_iters#3
    x_size = options.L*options.N_BINS #4
    print(x_size)
    lr = options.lr
    batch_size = options.batch_size

    # build net
    # create placeholders for input X (stft part) and mask y  #prob

    model = build_model(x_size,options.re)

    model.compile(loss='mean_squared_error', optimizer='sgd')
    #----------------------------------
    data_class = DataClass(options.train_mixes_filename, options.train_masks_filename, batch_size)

    loss_vec = np.zeros([n_iters, 1])

    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)


    for i in range(0, n_iters):
            print(i)
            batch = data_class.get_batch()
            model.train_on_batch(batch[0], batch[1])


            score = model.evaluate(batch[0], batch[1], verbose=0)
            print(score)
            loss_vec[i] = score
    #         save check point



    lossfilename = 'loss.pckl'
    f = open(lossfilename, 'wb')
    pickle.dump(loss_vec, f)
    f.close()

# serialize model to JSON
model_json = model.to_json()
with open(options.model_dir+"model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(options.model_dir+"model.h5")
print("Saved model to disk")
# _________________________________________________________________________________
# load json and create model
json_file = open(options.model_dir+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(options.model_dir+"model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='sgd')

probs = loaded_model.predict(mix_rowvecs)
