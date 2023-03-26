

from autograd import numpy as np
from autograd import value_and_grad

from tqdm import tqdm

import matplotlib.pyplot as plt

from homework_3.hw3 import gradient_descent


x = np.loadtxt('2d_span_data_centered.csv', delimiter=',')

def find_best_weights(weight_history, cost_history):
    weight_history = np.array(weight_history)
    cost_history = np.array(cost_history)

    best_cost = np.argmin(cost_history)

    return weight_history[best_cost]

def model(x_p, C):
    projection_matrix = C*C.T
    return np.dot(projection_matrix, x_p)


def autoencoder(C):
    cost = 0

    for i in range(x.shape[1]):
        cost += (model(x[:,i], C) - x[:,i]) ** 2

    return np.linalg.norm(cost) / x.shape[1]

initial_C = np.array([[-3.5],[3.5]])
weight_history, cost_history = gradient_descent(g=autoencoder, alpha=10**(-4), max_its=1000, w=initial_C)

best_weight = find_best_weights(weight_history, cost_history).flatten()

encoded_data = []

for i in range(x.shape[1]):
    encoded_data_i = np.dot(x[:,i], best_weight)
    encoded_data.append(encoded_data_i)

decoded_data_x = []
decoded_data_y = []

for x_p in encoded_data:
    decoded_data = np.dot(best_weight, x_p)
    decoded_data_x.append(decoded_data[0])
    decoded_data_y.append(decoded_data[1])

print(best_weight)
fig, axs = plt.subplots(2,2)
fig.set_size_inches(8, 8)
axs[0][0].plot(cost_history)
axs[0][1].scatter(x[0,:], x[1,:])
x_values = [0, best_weight[0]]
y_values = [0, best_weight[1]]
axs[0][1].arrow(0, 0, best_weight[0], best_weight[1], width = 0.1, head_width = 0.2)
axs[1][0].scatter(encoded_data, np.zeros(len(encoded_data)))
axs[1][1].scatter(decoded_data_x, decoded_data_y)
plt.show()