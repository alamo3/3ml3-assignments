import autograd.numpy as np
from autograd import grad, value_and_grad
from tqdm import tqdm
import matplotlib.pyplot as plt

def feature_transforms(x, w):
    a = w[0] + np.dot(x.T, w[1:])
    return np.sin(a).T


def gradient_descent(g,alpha,max_its,w, x,y):
    # compute gradient module using autograd

    # run the gradient descent loop
    weight_history = [w]     # container for weight history
    cost_history = [g(w,x,y)]    # container for corresponding cost function history

    gradient = value_and_grad(g)
    for k in tqdm(range(max_its)):
        # evaluate the gradient, store current weights and cost function value
        cost_eval, grad_eval = gradient(w,x,y)

        # take gradient descent step
        w = w - alpha*grad_eval

        # record weight and cost
        weight_history.append(w)
        cost_history.append(cost_eval)
    return weight_history, cost_history

def model(x, w):
    # feature transformation
    f = feature_transforms(x, w[0])

    # compute linear combination and return
    a = w[1][0] + np.dot(f.T, w[1][1:])
    return a.T


def load_data():
    csvname = 'multiple_sine_waves.csv'
    data = np.loadtxt(csvname, delimiter=',')
    x = data[:2, :]
    y = data[2:, :]

    return x,y

def least_squares(w, x, y):

    model_preds = model(x, w)

    cost_sum = 0

    for i in range(y.shape[1]):
        difference = model_preds[:,i] - y[:,i]
        cost_sum += np.linalg.norm(difference) ** 2

    return cost_sum / y.shape[1]



def train():
    x, y = load_data()

    w_init = np.random.randn(2,3,2)
    weight_history, cost_history = gradient_descent(least_squares, max_its=1000, alpha=0.01, w=w_init, x=x, y=y)

    plt.plot(cost_history)
    plt.show()



if __name__ == "__main__":
    train()