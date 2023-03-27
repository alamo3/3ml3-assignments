import numpy as np
from urllib.request import urlopen

import matplotlib.pyplot as plt




def load_dataset():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication'
    raw_data = urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",")
    print(dataset.shape)

    X = dataset[:, [1, 3]]
    Y = dataset[:, 4]

    return X, Y


if __name__ == "__main__":
    x, y = load_dataset()

    print(x.shape)
    print(y.shape)

    plt.plot(x.T[:-1,:], x.T[-1:,:], c=int(y))
    plt.show()
