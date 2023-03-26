import math

import tensorflow as tf
import tensorflow.python.data
from autograd import numpy as np
from autograd import value_and_grad
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels.shape=(1,60000)
test_labels.shape=(1,10000)

train_images=(train_images.reshape(60000,784)).T
test_images=(test_images.reshape(10000,784)).T

x_batch = None
y_batch = None


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, lst.shape[1] , n):
        upper = i+n
        if i + n >= lst.shape[1]:
            upper = lst.shape[1]-1
        yield lst[:,i:upper]


def gradient_descent(g,w,x_train,y_train,alpha,max_its, batch_size):
    # compute gradient module using autograd

    global x_batch, y_batch
    # run the gradient descent loop
    weight_history = [w]     # container for weight history
   # cost_history = [g(w)]    # container for corresponding cost function history

    x_train_batches = list(chunks(x_train, math.ceil(x_train.shape[1] / batch_size)))
    y_train_batches = list(chunks(y_train, math.ceil(y_train.shape[1] / batch_size)))
    cost_history = [g(w, x_train_batches[0], y_train_batches[0][0], 0)]

    gradient = value_and_grad(g)
    for k in tqdm(range(max_its)):
        cost_history_epoch = []
        weight_history_epoch = []
        for j in range(len(x_train_batches)):
            x_batch = x_train_batches[j]
            y_batch = y_train_batches[j]
            # evaluate the gradient, store current weights and cost function value
            cost_eval, grad_eval = gradient(w,x_batch, y_batch, j)

            # take gradient descent step
            w = w - alpha*grad_eval

            # record weight and cost
            weight_history_epoch.append(w)
            cost_history_epoch.append(cost_eval)

        weight_history.append(weight_history[-1])
        cost_history.append(cost_history_epoch[-1])

    return weight_history, cost_history

def model(x,w):
    a = w[0] + np.dot(x.T,w[1:])
    return a.T

# multiclass perceptron
def multiclass_perceptron(w,x,y,iter):
    # get subset of points
    x_p = x
    y_p = y
    # pre-compute predictions on all points
    all_evals = model(x_p,w)
    # compute maximum across data points
    a = np.max(all_evals,axis = 0)
    # compute cost in compact form using numpy broadcasting
    b = all_evals[y_p.astype(int).flatten(),np.arange(np.size(y_p))]
    cost = np.sum(a - b)

    # return average
    return cost/float(np.size(y_p))


w = 0.1*np.random.randn(784+1, 10)

weight_history, cost_history = gradient_descent(g=multiclass_perceptron, w=w, x_train=train_images, y_train=train_labels, alpha=0.001, max_its=5, batch_size=200)

plt.plot(cost_history)
plt.show()



