import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

VERTICAL = 0
HORIZONTAL = 1
def generate_stripe_image(size, stripe_nr, vertical = True):
  img=np.zeros((size,size,1),dtype="uint8")
  for i in range(0,stripe_nr):
    x,y = np.random.randint(0,size,2)
    l  = np.int(np.random.randint(y,size,1))
    if (vertical):
      img[y:l,x,0]=255
    else:
      img[x,y:l,0]=255
  return img

def test_generate_img():
    img = generate_stripe_image(50, 10, vertical=True)
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.waitforbuttonpress()



def create_training_validation_set(vertical, num_images):
    X_train = []
    X_val = []
    Y_train = []
    Y_val = []

    # generate images
    for i in range(num_images):
        img_train = generate_stripe_image(50,10, vertical=vertical)
        img_val = generate_stripe_image(50, 10, vertical=vertical)

        label = VERTICAL if vertical else HORIZONTAL

        X_train.append(img_train)
        X_val.append(img_val)
        Y_val.append(label)
        Y_train.append(label)


    return X_train, X_val, Y_train, Y_val


def train_model():
    X_train, X_val, Y_train, Y_val = create_training_validation_set(vertical=True, num_images=500)
    X_train_2, X_val_2, Y_train_2, Y_val_2 = create_training_validation_set(vertical=False, num_images=500)

    X_train.extend(X_train_2)
    X_val.extend(X_val_2)
    Y_train.extend(Y_train_2)
    Y_val.extend(Y_val_2)

    Y_train_hot_encoded = tf.keras.utils.to_categorical(Y_train, num_classes=2)
    Y_val_hot_encoded = tf.keras.utils.to_categorical(Y_val, num_classes=2)

    X_train = np.asarray(X_train, dtype=float) / 255.0
    X_val = np.asarray(X_val, dtype=float) / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(kernel_size=(5,5), activation='linear',input_shape=(50, 50, 1), padding='same', filters=1),
        tf.keras.layers.MaxPool2D(pool_size=50),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2,activation='softmax', name='dense_out')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, Y_train_hot_encoded,
                        validation_data=(X_val, Y_val_hot_encoded),
                        batch_size=64,
                        epochs=50,
                        verbose=1,
                        shuffle=True)

    print(model.count_params())
    print(model.summary())

    fig, axs = plt.subplots(2)

    kernel = model.get_weights()[0].squeeze()

    axs[0].plot(history.history['val_loss'])
    axs[1].imshow(kernel, cmap='gray')

    plt.show()
    plt.waitforbuttonpress()



if __name__ == "__main__":
    train_model()
