import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import *

x_breast_cancer = []
y_breast_cancer = []


def load_breast_cancer_data(file_path):
    # data input
    data1 = np.loadtxt(file_path, delimiter=',')

    global x_breast_cancer
    global y_breast_cancer
    # get input and output of dataset
    x = data1[:-1, :]
    y = data1[-1:, :]

    x_breast_cancer = x
    y_breast_cancer = y

def gradient_descent(g,alpha,max_its,w):
    # compute gradient module using autograd

    # run the gradient descent loop
    weight_history = [w]     # container for weight history
    cost_history = [g(w)]    # container for corresponding cost function history

    gradient = value_and_grad(g)
    for k in range(max_its):
        # evaluate the gradient, store current weights and cost function value
        cost_eval, grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(cost_eval)
    return weight_history, cost_history


def model(x,w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T

def sigmoid(t):
    return np.tanh(t)

def softmax(w):
    # compute the least squares cost
    cost = np.sum(np.log(1 + np.exp(-y_breast_cancer*(model(x_breast_cancer, w)))))
    return cost/float(np.size(y_breast_cancer))

def perceptron(w):

    sum_perceptron = np.sum(np.maximum(0, -y_breast_cancer*model(x_breast_cancer, w)))

    return sum_perceptron / float(np.size(y_breast_cancer))

def miscount(w, x, y):
    misclassifications = 0

    for i in range(np.size(y)):
        y_pred = -1 if model(x[:,i],w)[0] < 0 else 1
        y_actual = y[0][i]

        if y_pred != y_actual:
            misclassifications +=1

    return misclassifications

def exercise_1():
    load_breast_cancer_data('breast_cancer_data.csv')

    initial_w_softmax = 0.1*np.random.randn(9,1)
    initial_w_perception = 0.1*np.random.randn(9,1)

    weight_history_softmax, cost_history_softmax = gradient_descent(softmax, 1.0, 1000, initial_w_softmax)
    weight_history_perceptron, cost_history_perceptron = gradient_descent(perceptron, 0.1, 1000, initial_w_perception)
    miscount_history_softmax = [miscount(v, x_breast_cancer, y_breast_cancer) for v in weight_history_softmax]
    miscount_history_perceptron = [miscount(v, x_breast_cancer, y_breast_cancer) for v in weight_history_perceptron]
    print('Misclassifications softmax', miscount_history_softmax[-1])
    print('Misclassification perceptron', miscount_history_perceptron[-1])
    fig, axs = plt.subplots(2, 2)
    axs[0][0].plot(cost_history_softmax)
    axs[1][0].scatter([x for x in range(len(weight_history_softmax))],miscount_history_softmax)
    axs[1][1].scatter([x for x in range(len(weight_history_perceptron))], miscount_history_perceptron)
    axs[0][1].plot(cost_history_perceptron)

    plt.show()


if __name__ == "__main__":
    exercise_1()
