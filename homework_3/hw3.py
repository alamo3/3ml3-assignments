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

def softmax(w):
    # compute the least squares cost
    cost = np.sum(np.log(1 + np.exp(-y_breast_cancer*(model(x_breast_cancer, w)))))
    return cost/float(np.size(y_breast_cancer))

def perceptron(w):

    sum_perceptron = np.sum(np.maximum(0, -y_breast_cancer*model(x_breast_cancer, w)))

    return sum_perceptron / float(np.size(y_breast_cancer))

def miscount(w, x, y, malignant_only = False):
    misclassifications = 0

    for i in range(np.size(y)):
        y_pred = -1 if model(x[:,i],w)[0] < 0 else 1
        y_actual = y[0][i]

        if y_actual == 1 and malignant_only:
            continue

        if y_pred != y_actual:
            misclassifications +=1

    return misclassifications

def exercise_1_1_to_4():
    load_breast_cancer_data('breast_cancer_data.csv')

    initial_w_softmax = 0.1*np.random.randn(9,1)
    initial_w_perception = 0.1*np.random.randn(9,1)

    weight_history_softmax, cost_history_softmax = gradient_descent(softmax, 1.0, 1000, initial_w_softmax)
    weight_history_perceptron, cost_history_perceptron = gradient_descent(perceptron, 0.1, 1000, initial_w_perception)
    miscount_history_softmax = [miscount(v, x_breast_cancer, y_breast_cancer) for v in weight_history_softmax]
    miscount_history_perceptron = [miscount(v, x_breast_cancer, y_breast_cancer) for v in weight_history_perceptron]
    miscount_history_softmax_malignant = [miscount(v, x_breast_cancer, y_breast_cancer, malignant_only=True) for v in weight_history_softmax]
    miscount_history_perceptron_malignant = [miscount(v, x_breast_cancer, y_breast_cancer, malignant_only=True) for v in weight_history_perceptron]
    print('Misclassifications softmax', miscount_history_softmax_malignant[-1])
    print('Misclassification perceptron', miscount_history_perceptron_malignant[-1])
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(18, 20)
    axs[0][0].plot(cost_history_softmax)
    axs[0][1].plot(cost_history_perceptron)
    axs[1][0].scatter([x for x in range(len(weight_history_softmax))],miscount_history_softmax)
    axs[1][1].scatter([x for x in range(len(weight_history_perceptron))], miscount_history_perceptron)
    axs[2][0].scatter([x for x in range(len(weight_history_softmax))], miscount_history_softmax_malignant)
    axs[2][1].scatter([x for x in range(len(weight_history_perceptron))], miscount_history_perceptron_malignant)

    plt.show()

def sigmoid(t):
    return 1/(1 + np.exp(-t))


yc = []
# the convex cross-entropy cost function
lam = 2*10**(-3)
def cross_entropy(w):
    # compute sigmoid of model
    a = sigmoid(model(x_breast_cancer,w))
    # compute cost of label 0 points
    ind = np.argwhere(yc == 0)
    cost = -np.sum(np.log(1 - a[:,ind]))
    # add cost on label 1 points
    ind = np.argwhere(yc==1)
    cost -= np.sum(np.log(a[:,ind]))
    # add regularizer
    cost += lam*np.sum(w[1:]**2)
    # compute cross-entropy
    return cost/float(np.size(yc))


def misclassification_logistic(w, x, y, malignant_only = False):
    misclassifications = 0

    for i in range(np.size(y)):
        y_pred = 0 if sigmoid(model(x[:, i], w)[0]) < 0.5 else 1
        y_actual = y[i]

        if y_actual == 1 and malignant_only:
            continue

        if y_pred != y_actual:
            misclassifications += 1

    return misclassifications

def ex_1_5():
    global yc
    load_breast_cancer_data('breast_cancer_data.csv')

    a = np.argwhere(y_breast_cancer > 0.9)
    b = np.argwhere(y_breast_cancer < -0.9)
    yc = np.arange(699)
    yc[a] = 1
    yc[b] = 0

    initial_w = 0.1*np.random.randn(9,1)

    weight_history, cost_history = gradient_descent(cross_entropy,0.6,1000,initial_w)
    miscount_history_logistic = [misclassification_logistic(v, x_breast_cancer, yc) for v in weight_history]
    miscount_history_logistic_malignant = [misclassification_logistic(v, x_breast_cancer, yc, malignant_only=True) for v in weight_history]

    print('Number of malignant misclassified', min(miscount_history_logistic_malignant))

    fig, axs = plt.subplots(3)
    fig.set_size_inches(10, 15)
    axs[0].plot(cost_history)
    axs[1].scatter([x for x in range(len(weight_history))], miscount_history_logistic)
    axs[2].scatter([x for x in range(len(weight_history))], miscount_history_logistic_malignant)

    plt.show()

def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.nanmean(x,axis = 1)[:,np.newaxis]
    x_stds = np.nanstd(x,axis = 1)[:,np.newaxis]
    # check to make sure that x_stds > small threshold, for those not
    # divide by 1 instead of original standard deviation
    ind = np.argwhere(x_stds < 10**(-2))
    if len(ind) > 0:
        ind = [v[0] for v in ind] # Just keep the row index
        adjust = np.zeros((x_stds.shape))
        adjust[ind] = 1.0
        x_stds += adjust
    # fill in any nan values with means
    ind = np.argwhere(np.isnan(x) == True)
    for i in ind:
        x[i[0],i[1]] = x_means[i[0]]
    # create standard normalizer function
    normalizer = lambda data: (data - x_means)/x_stds
    # create inverse standard normalizer
    inverse_normalizer = lambda data: data*x_stds + x_means
    # return normalizer
    return normalizer,inverse_normalizer

x_spam = []
y_spam = []

def softmax_spam(w):
    # compute the least squares cost
    cost = np.sum(np.log(1 + np.exp(-y_spam*(model(x_spam, w)))))
    return cost/float(np.size(y_spam))

def perceptron_spam(w):

    sum_perceptron = np.sum(np.maximum(0, -y_spam*model(x_spam, w)))

    return sum_perceptron / float(np.size(y_spam))

def load_spam_database():
    data1 = np.loadtxt('spambase_data.csv', delimiter=',')

    global x_spam, y_spam

    x_spam = data1[:-1,:]
    y_spam = data1[-1:,:]

def miscount_linear_boundary(w, x, y):

    miscounts = 0

    for i in range(np.size(y)):
        y_pred = -1 if model(x[:,i], w) < 0 else 1
        y_actual = y[0][i]

        if y_pred != y_actual:
            miscounts += 1

    return miscounts


def ex_2():
    load_spam_database()

    global x_spam, y_spam

    print("Spam database loaded!")

    x_normalizer, x_inverse_normalizer = standard_normalizer(x_spam)
    y_normalizer, y_inverse_normalizer = standard_normalizer(y_spam)

    x_spam = x_normalizer(x_spam)

    w_initial_softmax = 0.1*np.random.randn(57+1,1)
    w_initial_perceptron = 0.1*np.random.randn(57+1,1)

    weight_history_softmax, cost_history_softmax = gradient_descent(softmax_spam, 1.0, 3000, w_initial_softmax)
    weight_history_perceptron, cost_history_perceptron = gradient_descent(perceptron_spam, 0.1, 3000, w_initial_perceptron)

    miscount_history_softmax = [miscount_linear_boundary(v, x_spam, y_spam) for v in weight_history_softmax]
    miscount_history_perceptron = [miscount_linear_boundary(v, x_spam, y_spam) for v in weight_history_perceptron]

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(15,15)
    axs[0][0].plot(cost_history_softmax)
    axs[0][1].plot(cost_history_perceptron)
    axs[1][0].scatter([x for x in range(len(weight_history_softmax))], miscount_history_softmax)
    axs[1][1].scatter([x for x in range(len(weight_history_perceptron))], miscount_history_perceptron)

    plt.show()

    miscount_history_softmax = np.array(miscount_history_softmax)
    min_miscount = np.argmin(miscount_history_softmax)
    print('Lowest misclassifications softmax:', miscount_history_softmax[min_miscount])
    print('Highest accuracy softmax: ', 1 - miscount_history_softmax[min_miscount] / np.size(y_spam))

    miscount_history_perceptron = np.array(miscount_history_perceptron)
    min_miscount = np.argmin(miscount_history_perceptron)
    print('Lowest misclassifications perceptron: ', miscount_history_perceptron[min_miscount])
    print('Highest accuracy perceptron: ', 1 - miscount_history_perceptron[min_miscount] / np.size(y_spam))


x_credit = []
y_credit = []

def load_credit_dataset():
    data1 = np.loadtxt('credit_dataset.csv', delimiter=',')

    global x_credit, y_credit

    x_credit = data1[:-1,:]
    y_credit = data1[-1:,:]

def perceptron_credit(w):

    sum_perceptron = np.sum(np.maximum(0, -y_credit*model(x_credit, w)))

    return sum_perceptron / float(np.size(y_credit))

def ex_3():
    load_credit_dataset()

    global x_credit, y_credit
    x_credit_normalizer, x_credit_norm_inv = standard_normalizer(x_credit)

    x_credit = x_credit_normalizer(x_credit)

    w_initial_perceptron = 0.1 * np.random.randn(20 + 1, 1)

    weight_history, cost_history = gradient_descent(perceptron_credit, 0.1, 1000, w_initial_perceptron)

    miscount_history = [miscount_linear_boundary(v,x_credit,y_credit) for v in weight_history]

    fig, axs=plt.subplots(2)
    axs[0].plot(cost_history)
    axs[1].scatter([x for x in range(len(weight_history))], miscount_history)

    plt.show()

    miscount_history = np.array(miscount_history)

    misclass_min = np.argmin(miscount_history)

    print('Accuracy of perceptron cost: ', 1 - miscount_history[misclass_min]/np.size(y_credit))


x_toy = []
y_toy = []

lam_2 = 10**-5 # our regularization paramter
def multiclass_perceptron(w):
    # pre-compute predictions on all points
    all_evals = model(x_toy,w)
    # compute maximum across data points
    a = np.max(all_evals, axis = 0)
    # compute cost in compact form using numpy broadcasting
    b = all_evals[y_toy.astype(int).flatten(),np.arange(np.size(y_toy))]
    cost = np.sum(a - b)
    # add regularizer
    cost = cost + lam_2*np.linalg.norm(w[1:,:],'fro')**2
    # return average
    return cost/float(np.size(y_toy))

def load_toy_dataset():
    data1 = np.loadtxt('3class_data.csv', delimiter=',')
    global x_toy, y_toy

    x_toy = data1[:-1,:]
    y_toy = data1[-1:,:]

def plot_decision_boundary(coefficients):
    b = coefficients[0]
    x1 = coefficients[1]
    x2 = coefficients[2]


def ex_4():

    load_toy_dataset()

    global x_toy,y_toy
    x_toy_normalizer,_ = standard_normalizer(x_toy)

#    x_toy = x_toy_normalizer(x_toy)

    w = 0.1 * np.random.randn(3, 3)

    weight_history, cost_history = gradient_descent(multiclass_perceptron, alpha=0.1, max_its=1000, w=w)

    fig, axs = plt.subplots(2)
    axs[0].plot(cost_history)
    axs[1].scatter(x_toy[:-1,:], x_toy[-1:,:], c=y_toy)
    plt.show()




if __name__ == "__main__":
    ex_4()
