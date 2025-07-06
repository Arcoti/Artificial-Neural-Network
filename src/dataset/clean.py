import numpy as np

def normalize(ds):
    return np.array([(image.astype(np.float32) / 255.0, label) for image, label in ds])

def shuffle_and_batch(ds, batch_size: int = 32):
    # Shuffle
    np.random.shuffle(ds)
    
    # Batching
    return np.array([ds[start, start + batch_size] for start in range(0, ds.shape[0], batch_size)])

def flatten(ds):
    return np.array([(image.reshape(image.shape[0]*image.shape[1], 1), label) for image, label in ds])

def separate(ds, label_size: int = 10, as_label: bool = False):
    X = []
    Y = []
    for batch in ds:
        batch_X = []
        batch_Y = []
        for image, label in batch:
            # Get X
            batch_X.append(image)

            # Get Y
            if as_label:
                batch_Y.append(label)
            else:
                one_hot = np.zeros(label_size) # 10 digits only
                one_hot[label] = 1
                batch_Y.append(one_hot)

        X.append(batch_X)
        Y.append(batch_Y)
    
    return np.array(X), np.array(Y)

def clean(ds):
    normalized_ds = normalize(ds)
    flattened_ds = flatten(normalized_ds)
    batched_ds = shuffle_and_batch(flattened_ds)
    return separate(batched_ds)