import keras
import numpy as np
from urllib.request import urlopen

import matplotlib.pyplot as plt
import tensorflow as tf




def load_dataset():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication'
    raw_data = urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",")
    print(dataset.shape)

    X = dataset[:, [1, 3]]
    Y = dataset[:, 4]

    return X, Y

def create_train_model(x, y):

    model = keras.Sequential([tf.keras.layers.Dense(1, batch_input_shape=(None, 2), activation='sigmoid', name='dense')])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.15), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    history = model.fit(x, y, epochs=400, batch_size=128)

    fix, axs = plt.subplots(2)
    axs[0].plot(history.history['loss'])
    axs[1].plot(history.history['mean_squared_error'])
    plt.show()
    print(history.params)

    x1list, x2list =  np.meshgrid(np.linspace(np.min(x[:, 0]) - 2, np.max(x[:, 0]) + 2, 50),
                                  np.linspace(np.min(x[:, 1]) - 2, np.max(x[:, 1]) + 2, 50))
    x_test = np.array([x1list.ravel(), x2list.ravel()])
    y_pred = model.predict(x_test.T)

    x1list_T = np.linspace(np.min(x[:, 0]) - 2, np.max(x[:, 0]) + 2, 50)
    x2list = np.linspace(np.min(x[:, 1]) - 2, np.max(x[:, 1]) + 2, 50)

    plt.contourf(x1list_T, x2list, y_pred.reshape(x1list.shape))
    colors = y.astype(int).tolist()

    plt.scatter(x.T[:-1, :][0], x.T[-1:, :][0], c=colors)
    plt.show()

def create_train_dense_model(x, y):
    model = keras.Sequential(
        [tf.keras.layers.Dense(8, batch_input_shape=(None, 2), activation='sigmoid', name='dense'),
         tf.keras.layers.Dense(2,activation='softmax', name='dense_out')])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.15), loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    Y_hot = tf.keras.utils.to_categorical(y, 2)
    history = model.fit(x, Y_hot, epochs=400, batch_size=128)

    fig, axs = plt.subplots(2)
    axs[0].plot(history.history['loss'])
    axs[1].plot(history.history['mean_squared_error'])
    plt.show()

    x1list, x2list = np.meshgrid(np.linspace(np.min(x[:, 0]) - 2, np.max(x[:, 0]) + 2, 50),
                                 np.linspace(np.min(x[:, 1]) - 2, np.max(x[:, 1]) + 2, 50))
    x_test = np.array([x1list.ravel(), x2list.ravel()])
    y_pred = model.predict(x_test.T)

    y_real = y_pred[:,1]

    x1list_T = np.linspace(np.min(x[:, 0]) - 2, np.max(x[:, 0]) + 2, 50)
    x2list = np.linspace(np.min(x[:, 1]) - 2, np.max(x[:, 1]) + 2, 50)

    plt.contourf(x1list_T, x2list, y_real.reshape(x1list.shape))
    colors = y.astype(int).tolist()

    plt.scatter(x.T[:-1, :][0], x.T[-1:, :][0], c=colors)
    plt.show()



if __name__ == "__main__":
    x, y = load_dataset()

    print(x.shape)
    print(y.shape)


    create_train_dense_model(x, y)
