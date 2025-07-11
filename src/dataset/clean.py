import numpy as np

def normalize(ds):
    return [(image.astype(np.float32) / 255.0, label) for image, label in ds]

def shuffle_and_batch(ds, batch_size):
    # Shuffle
    np.random.shuffle(ds)
    
    # Batching
    return [ds[start: start + batch_size] for start in range(0, len(ds), batch_size)]

def flatten(ds):
    return [(image.reshape(image.shape[0]*image.shape[1]), label) for image, label in ds]

def separate(ds, label_size: int, as_label: bool):
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

        X.append(np.array(batch_X).T)
        Y.append(np.array(batch_Y).T)
    
    return X, Y

def clean(ds, as_label: bool = False, label_size: int = 10, batch_size: int = 32):
    flattened = flatten(ds)
    normalized = normalize(flattened)
    batched = shuffle_and_batch(normalized, batch_size)
    return separate(batched, label_size, as_label)