import tensorflow_datasets as tfds

def retrieve_mnist():
    ds_train_raw = tfds.load('mnist', split='train', shuffle_files=True, as_supervised=True)
    ds_test_raw = tfds.load('mnist', split='test', shuffle_files=True, as_supervised=True)

    ds_train = tfds.as_numpy(ds_train_raw)
    ds_test = tfds.as_numpy(ds_test_raw)

    # ds_train = image, label
    # image is 28 x 28 (one array is one pixel)
    # label is the answer

    print(len(ds_train), len(ds_test))

    return ds_train, ds_test

retrieve_mnist()