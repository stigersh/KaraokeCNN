import numpy as np
import pandas as pd


def chunck_generator(filename,chunk_size):
    for chunk in pd.read_csv(filename, header=None, chunksize=chunk_size):
        yield (chunk.as_matrix())


class DataClass:
    def __init__(self, mixes_filename, masks_filename, batch_size):
        self.mixes_filename = mixes_filename
        self.masks_filename = masks_filename
        self.batchsize = batch_size
        self.generator_train_in = chunck_generator(self.mixes_filename,self.batchsize)
        self.generator_train_out = chunck_generator(self.masks_filename,self.batchsize)

    def get_batch(self):
        try:
            batch = [next(self.generator_train_in) , next(self.generator_train_out)]
        except:
            batch = []
        return batch
