from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255



def train_neural_network(learning_rate):

    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

    return history

def ex2():
    model = keras.Sequential([layers.Dense(10, activation='softmax')])
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history_small_model = model.fit(
        train_images, train_labels,
        epochs=50,
        batch_size=128,
        validation_split=0.2)

    plt.plot(history_small_model.history['val_loss'])
    plt.show()

def ex3():
    model = keras.Sequential([
        layers.Dense(96, activation='relu'),
        layers.Dense(96, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history_large_model = model.fit(
        train_images, train_labels,
        epochs=50,
        batch_size=128,
        validation_split=0.2)

    plt.plot(history_large_model.history['val_loss'])
    plt.show()

def ex1():

    learning_rates = [1, 0.5, 0.01, 0.001]

    histories = [train_neural_network(lr) for lr in learning_rates]

    fig, axs = plt.subplots(4)

    for i in range(len(learning_rates)):
        axs[i].plot(histories[i].history['val_loss'])

    plt.show()


if __name__ == "__main__":
    ex3()


