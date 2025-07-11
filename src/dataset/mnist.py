import numpy as np
import tensorflow_datasets as tfds

def retrieve_mnist():
    ds_train_raw = tfds.load('mnist', split='train', shuffle_files=True, as_supervised=True)
    ds_test_raw = tfds.load('mnist', split='test', shuffle_files=True, as_supervised=True)

    ds_train = tfds.as_numpy(ds_train_raw)
    ds_test = tfds.as_numpy(ds_test_raw)

    return ds_train, ds_test
