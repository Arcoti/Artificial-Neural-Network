import matplotlib.pyplot as plt

def learning_rate_finder(step: int, total: int, start=1e-3, end=1):
    return start * (end / start) ** (step / (total - 1))

def plot(x, y, x_label: str, y_label: str, title: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()