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

def separate(ds, label_size: int):
    X = []
    Y = []
    for batch in ds:
        batch_X = []
        batch_Y = []
        for image, label in batch:
            # Get X
            batch_X.append(image)

            # Get Y
            one_hot = np.zeros(label_size) # 10 digits only
            one_hot[label] = 1
            batch_Y.append(one_hot)

        X.append(np.array(batch_X))
        Y.append(np.array(batch_Y))
    
    return X, Y

def clean_train(ds, batch_size: int, label_size: int):
    flattened = flatten(ds)
    normalized = normalize(flattened)
    batched = shuffle_and_batch(normalized, batch_size)
    return separate(batched, label_size)

def clean_test(ds, batch_size: int):
    flattened = flatten(ds)
    normalized = normalize(flattened)
    batched = shuffle_and_batch(normalized, batch_size)

    X = []
    Y = []

    for batch in batched:
        images = []
        labels = []
        for image, label in batch:
            images.append(image)
            labels.append(label)
        X.append(np.array(images))
        Y.append(np.array(labels))
    return X, Y