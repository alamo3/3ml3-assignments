# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(6.3 * rng.rand(100, 1), axis=0)
y = np.sin(X).ravel()
y += 0.3 * (0.5 - rng.rand(len(X)))

yreg = []
def sklearn_tree_regressor_test():
    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)

    return y_1, y_2

def plot_results(y_1, y_2, y_our):
    # Plot the results
    plt.figure()

    if y_our is not None:
        plt.plot(X, y_our, color="red", label="My Regression Tree")

    plt.scatter(X, y, s=20, edgecolor="black",
                c="darkorange", label="data")
    plt.plot(X, y_1, color="cornflowerblue",
             label="maxdepth=2", linewidth=2)
    plt.plot(X, y_2, color="yellowgreen", label="maxdepth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()


def My_TreeRegressor(y,il,ir,max_depth,level=0):
    global yreg
    if (level==0): yreg=np.zeros(len(y))

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

    My_TreeRegressor(y,il,indices_left[-1],max_depth,level+1)
    My_TreeRegressor(y,indices_right[0],ir,max_depth,level+1)





if __name__ == "__main__":
    y1, y2 = sklearn_tree_regressor_test()
    X = X.squeeze()
    My_TreeRegressor(y,0,len(y)-1,10)
    plot_results(y1, y2, yreg)

