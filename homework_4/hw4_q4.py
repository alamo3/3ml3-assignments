import matplotlib.pyplot as plt
import numpy as np

def center(X):
    X_means = np.mean(X,axis=1)[:,np.newaxis]
    X_normalized = X - X_means

    return X_normalized

def compute_pcs(X,lam):
    # create the correlation matrix
    P = float(X.shape[1])
    Cov = 1/P*np.dot(X,X.T) + lam*np.eye(X.shape[0])

    # use numpy function to compute eigenvalues / vectors of correlation matrix
    D,V = np.linalg.eigh(Cov)
    return D,V


X_original = np.loadtxt('2d_span_data.csv',delimiter=',')

x_center = center(X_original)

D,V = compute_pcs(x_center, 10**(-5))

encoded_points = np.zeros(shape=(2,x_center.shape[1]))

projection_matrix = np.array([V[0], V[1]]).T
for i in range(x_center.shape[1]):
    encoded_point_i = np.dot(projection_matrix, x_center[:,i])
    encoded_points[0][i] = encoded_point_i[0]
    encoded_points[1][i] = encoded_point_i[1]

fig, axs = plt.subplots(2)
fig.set_size_inches(3.5, 6)
axs[0].scatter(x_center[0,:], x_center[1,:])
axs[0].arrow(0, 0, V[0][0], V[0][1], width = 0.1, head_width = 0.2)
axs[0].arrow(0, 0, V[1][0], V[1][1], width = 0.1, head_width = 0.2)
axs[1].scatter(encoded_points[0,:], encoded_points[1,:])


plt.show()