import math

import autograd.numpy.numpy_boxes
import matplotlib.pyplot as plt
import numpy as np
import pandas
from autograd import grad
from autograd import *

import pandas as pd

# gradient descent function - inputs: g (input function),
# alpha (steplength parameter),
# max_its (maximum number of iterations), w (initialization)
def gradient_descent_exponential_decay(g, alpha, max_its, w, beta):
    # compute gradient module using autograd
    gradient = value_and_grad(g)
    # run the gradient descent loop
    weight_history = [w] # container for weight history

    cost_eval_init, grad_eval_init = gradient(w)
    cost_history = [cost_eval_init] # container for corresponding cost function history
    derivative_history = [-grad_eval_init]
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        cost_eval, grad_eval = gradient(w)

        derivative_k_1 = beta * derivative_history[-1] + (1-beta) * grad_eval
        # take gradient descent step
        w = w - alpha*derivative_k_1
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
        derivative_history.append(derivative_k_1)

    return weight_history,cost_history




def exercise_1_2():
    C = np.array([[0.5, 0], [0, 9.75]])
    g = lambda w: np.dot(np.dot(w.T, C), w)

    weight_history, cost_history = gradient_descent_exponential_decay(g=g, alpha=0.1, max_its=25, w=np.array([10, 1], dtype=float), beta=0)
    weight_history_1 , cost_history_1 = gradient_descent_exponential_decay(g=g, alpha=0.1, max_its=25, w = np.array([10, 1], dtype=float), beta=0.1)
    weight_history_2, cost_history_2 = gradient_descent_exponential_decay(g=g, alpha=0.1, max_its=25, w=np.array([10, 1], dtype=float),
                                                                          beta=0.7)

    g_2 = lambda x,y: (0.5 * x**2) + (9.75 * y**2)

    v_g = np.vectorize(g_2)

    xlist = np.linspace(-2, 20, 100)
    ylist = np.linspace(-5.0, 5.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = v_g(X,Y)

    fig, axs = plt.subplots(2)
    axs[0].contour(X,Y,Z)
    axs[0].scatter([i[0] for i in weight_history], [i[1] for i in weight_history], marker='o')
    axs[0].scatter([i[0] for i in weight_history_1], [i[1] for i in weight_history_1], marker='^')
    axs[0].scatter([i[0] for i in weight_history_2], [i[1] for i in weight_history_2], marker='^')


    axs[1].plot(cost_history)


    plt.show()


def gradient_descent(g,alpha,max_its,w):
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w]     # container for weight history
    cost_history = [g_2_np(w)]    # container for corresponding cost function history
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g_2_np(w))
    return weight_history,cost_history


def gradient_descent_normalized(g,alpha,max_its,w):
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w]     # container for weight history
    cost_history = [g_2_np(w)]    # container for corresponding cost function history
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - (alpha / (np.linalg.norm(grad_eval) + np.array([0.0000001, 0.000001])))*grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g_2_np(w))
    return weight_history,cost_history


def g_2(w):
    val = autograd.numpy.tanh(autograd.numpy.radians(4 * w[0] + 4 * w[1])) + max(1, 0.4 * w[0] ** 2) + 1
    return val

def g_2_np(w_l):
    return math.tanh(4 * w_l[0] + 4 * w_l[1]) + max(1, 0.4 * w_l[0] ** 2) + 1

def exercise_2_1():

    g = g_2

    weight_history, cost_history = gradient_descent(g, 0.1, 1000, w=np.array([2, 2], dtype = float))
    weight_history_normalized, cost_history_normalized = gradient_descent_normalized(g, 0.1, 1000, w=np.array([2, 2], dtype=float))


    plt.plot(cost_history_normalized)
    plt.plot(cost_history, "r-")
    plt.show()



def exercise_3():
    data = np.asarray(pd.read_csv('student_debt_data.csv', header=None))

    X = data[:,0]
    X.shape = (len(X), 1)
    o = np.ones((len(X), 1))

    X_new = np.concatenate((o,X), axis=1)

    Y = data[:,1]

    A = np.zeros((2,2))

    for i in range(len(X)):
        A_add = np.matrix([[X_new[i].T[0]*X_new[i][0], X_new[i].T[0]*X_new[i][1]],[X_new[i].T[1]*X_new[i][0], X_new[i].T[1]*X_new[i][1]] ])
        A = A + A_add

    B = np.asarray([0, 0])

    for i in range(len(data)):
        B = B + (np.dot(X_new[i], Y[i]))

    A_inv = np.linalg.pinv(A)

    W = np.asarray(np.matmul(A_inv, B))[0]


    y_pred = np.matmul(X_new, W)

    plt.scatter(X, Y)
    plt.scatter(X, y_pred)
    plt.show()



P = 0

X_data = None
Y_data = None

def gradient_descent_normal(g,alpha,max_its,w):
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w]     # container for weight history
    cost_history = [g(w)]    # container for corresponding cost function history
    for k in range(max_its):
        print('iteration ', k)
        # evaluate the gradient, store current weights and cost function value
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history,cost_history

def model(w):
    return w[1]*X_data + w[0]



def least_squares(w):
    Y_pred = model(w)

    least_sum = 0

    global X_data
    global Y_data

    for i in range(P):
        least_sum += (Y_pred[i] - Y_data[i]) * (Y_pred[i] - Y_data[i])

    return (1/P) * least_sum

def exercise_4():
    data = np.asarray(pd.read_csv('kleibers_law_data.csv', header=None))

    X = data[0,:]
    Y = data[1,:]

    X = np.log(X)
    Y = np.log(Y)

    global P, X_data, Y_data
    P = len(X)

    X_data = X
    Y_data = Y

    w_init = np.asarray([0.1, 0.1], dtype=float)

    weight_history, cost_history = gradient_descent_normal(least_squares, 0.01, 250, w_init)

    final_weight = weight_history[-1]

    y_pred = model(final_weight)

    fig, axs = plt.subplots(2)
    axs[0].scatter(X_data, Y_data)
    axs[0].scatter(X_data, y_pred)
    axs[1].plot(cost_history)


    plt.show()
    pass


if __name__ == "__main__":
    exercise_3()




