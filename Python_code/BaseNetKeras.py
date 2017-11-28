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
    model.add(Dense(x_size,input_shape=(x_size,),
                 activation='relu',kernel_regularizer=regularizers.l2(reg)))

    # model.add(Dense(x_size,
    #              activation='relu',kernel_regularizer=regularizers.l2(reg)))
    # model.add(Dense(x_size,
    #              activation='relu',kernel_regularizer=regularizers.l2(reg)))

    return model


def evaluate_epoch_error(model,mixes_filename,masks_filename,batch_size):
    data_class = DataClass(mixes_filename, masks_filename, batch_size)
    bnewEpoc = False
    tot_loss = 0
    counter = 0
    while bnewEpoc==False :
        batch, bnewEpoc = data_class.get_batch()
        if bnewEpoc:
            break
        counter+=1
        tot_loss += model.test_on_batch(batch[0], batch[1])
    tot_loss /= counter
    return tot_loss

def evaluate_over_n_first_batches(model,mixes_filename,masks_filename,batch_size,n):
    data_class = DataClass(mixes_filename, masks_filename, batch_size)
    bnewEpoc = False
    tot_loss = 0
    counter = 0
    while counter < n :
        batch, bnewEpoc = data_class.get_batch()
        if bnewEpoc:
            break
        counter+=1
        tot_loss += model.test_on_batch(batch[0], batch[1])
    tot_loss /= counter
    return tot_loss
#----------------------------------
if __name__ == "__main__":
    options = opt

    x_size = options.L*options.N_BINS #4
    print(x_size)
    lr = options.lr
    batch_size = options.batch_size

    # build net
    model = build_model(x_size,options.reg)

    model.compile(loss='mean_squared_error', optimizer='sgd')
    #----------------------------------
    data_class = DataClass(options.train_mixes_filename, options.train_masks_filename, batch_size)

    if not os.path.exists(options.model_dir+'/KERAS'):
        os.makedirs(options.model_dir+'/KERAS')


    n_epochs = 0
    i = 0
    loss_test = 0
    loss_vec = []
    valid_vec = []

    # serialize model to JSON
    model_json = model.to_json()
    with open(options.model_dir + "/KERAS/model.json", "w") as json_file:
        json_file.write(model_json)


    while n_epochs < options.n_epochs:
        print(i)
        batch, bnewEpoc = data_class.get_batch()
        if bnewEpoc:
            n_epochs += 1
        model.train_on_batch(batch[0], batch[1])

        if i % options.save_iters == 0:
            loss_train = evaluate_over_n_first_batches(model, options.train_mixes_filename,
                                                       options.train_masks_filename, batch_size, i + 1)
            loss_vec.append(loss_train)

            loss_valid = evaluate_epoch_error(model, options.valid_mixes_filename,
                                              options.valid_masks_filename, batch_size)
            valid_vec.append(loss_valid)

            print('step %d, loss train %g' % (i, loss_train))
            print('step %d, loss valid %g' % (i, loss_valid))

            # serialize weights to HDF5
            model.save_weights(options.model_dir + "/KERAS/weights_"+str(i)+".h5")

        i += 1
    model.save_weights(options.model_dir + "/KERAS/weights_final.h5")

    lossfilename = options.model_dir+'/KERAS/KerasparamsAndloss.pckl'
    f = open(lossfilename, 'wb')
    pickle.dump(loss_vec, f)
    pickle.dump(valid_vec, f)
    pickle.dump(options, f)
    pickle.dump(loss_test, f)
    f.close()

print("Saved model to disk")
# _________________________________________________________________________________

