import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets.samples_generator import (make_blobs, make_circles, make_moons)

# diabetes dataset
data = np.loadtxt('hw4_Diabetes_Normalized.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)
num_train = int(0.75 * n)

sample_train = data[0:num_train, 0:-1]
sample_test = data[num_train:, 0:-1]

label_train = data[0:num_train, -1]
label_test = data[num_train:, -1]

beta = np.random.normal(0, 0.01, size=(p - 1,))  # Initialize a random Beta using Gaussian Distribution
lam = 0.00001  # Lambda initialize
beta_updates = []  # Number of updates vector
err_matt = []  # Test Error Matrix
label_pred = np.copy(label_test)  # Predicted value


# The following function will calculate the P1 n-dimensional vector using the probability function
# to get the sigmoid probability function for the given lamda
def getP1(model, sample):
    temp = np.matmul(sample, model)
    result = 1 / (1 + np.exp(temp))
    # result = np.exp(-temp) / (1 + np.exp(-temp))
    return result


for i in range(100):
    temp1 = label_train - getP1(beta, sample_train)  # This is equals to Y-P1 on the maximum # likelihood function
    L_derivative = np.matmul(-sample_train.transpose(), temp1)  # This is the maximum likelihood function for beta

    if i != 0:
        beta = beta - lam*L_derivative
        label_pred = (np.matmul(sample_test, beta) > 0) * 1

    err = sum(abs(label_test - label_pred)) / (n - num_train)
    err_matt.append(err)
    beta_updates.append(i)

plt.plot(beta_updates[1:], err_matt[1:])
plt.xlabel('Beta Updates')
plt.ylabel('Testing Error')
plt.title('Testing Error vs Beta Updates')
plt.show()
