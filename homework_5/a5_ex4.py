# Import the necessary modules and libraries
import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X_sklearn = np.sort(6.3 * rng.rand(100, 1), axis=0)
X = X_sklearn.squeeze()
y = np.sin(X).ravel()
y += 0.3 * (0.5 - rng.rand(len(X)))

def sklearn_tree_regressor_test(max_depth):
    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=max_depth)
    regr_1.fit(X_sklearn, y)

    # Predict
    y_1 = regr_1.predict(X_sklearn)

    return y_1

def plot_results(axs, depth):
    # Plot the results

    y_sklearn = sklearn_tree_regressor_test(depth)

    yreg = np.zeros(len(y))
    My_TreeRegressor(y, 0, len(y)-1, depth, yreg)

    axs.scatter(X, y, s=20, edgecolor="black",
                c="darkorange", label="data")

    axs.plot(X, yreg, color="red", label="MyTree Maxdepth="+str(depth), linewidth=2)

    axs.plot(X, y_sklearn, color="yellowgreen", label="maxdepth="+str(depth), linewidth=2)

    axs.legend()

def comparison_sklearn_our(min_depth, max_depth):

    num_plots = max_depth - min_depth + 1


    fig, axs = plt.subplots(num_plots, figsize=(10, math.ceil(num_plots * 2.5)))
    fig.tight_layout()
    for i in range(num_plots):
        plot_results(axs[i], min_depth + i)


    plt.show()


def My_TreeRegressor(y, il, ir, max_depth, yreg, level=0):

    if (level==max_depth):
        yreg[il:ir]=np.mean(y[il:ir])
        return

    if abs(ir-il)==1:
        yreg[il:ir]=np.mean(y[il:ir])
        return

    if ir == il:
        yreg[ir] = y[ir]
        return

    midpoint_sk = (X[il] + X[ir]) / 2
    indices_left = []
    indices_right = []

    for i in range(il, ir+1):
        if (X[i] <= midpoint_sk):
            indices_left.append(i)

        if (X[i] > midpoint_sk):
            indices_right.append(i)

    vl = np.mean(y[indices_left])
    vr = np.mean(y[indices_right])
    yreg[indices_left] = vl
    yreg[indices_right] = vr

    My_TreeRegressor(y,il,indices_left[-1],max_depth,yreg,level+1)
    My_TreeRegressor(y,indices_right[0],ir,max_depth,yreg,level+1)


if __name__ == "__main__":
    comparison_sklearn_our(2, 6)

