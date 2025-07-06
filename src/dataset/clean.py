import numpy as np

def normalize(ds):
    return np.array([(image.astype(np.float32) / 255.0, label) for image, label in ds])

def shuffle_and_batch(ds, batch_size: int = 32):
    # Shuffle
    np.random.shuffle(ds)
    
    # Batching
    return [ds[start, start + batch_size] for start in range(0, ds.shape[0], batch_size)]

def flatten(ds):
    return np.array([(image.reshape(image.shape[0]*image.shape[1], 1), label) for image, label in ds])

def clean(ds):
    normalized_ds = normalize(ds)
    flattened_ds = flatten(normalized_ds)
    return shuffle_and_batch(flattened_ds)