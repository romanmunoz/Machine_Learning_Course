import numpy as np
import random
from matplotlib import pyplot as plt

# diabetes dataset
data = np.loadtxt('hw4_Diabetes_Normalized.csv', delimiter=',', skiprows=1)
[n, p] = np.shape(data)
num_train = int(0.75 * n)
adj = 0.05

sample_train = data[0:num_train, 0:-1]
sample_test = data[num_train:, 0:-1]

label_train = data[0:num_train, -1]
label_test = data[num_train:, -1]

beta = np.random.normal(0, 0.1, size=(p-1,))  # Initialize a random Beta using Gaussian Distribution
beta_updates = []  # Number of updates vector
err_matt = []  # Test Error Matrix
label_pred = np.copy(label_test)  # Predicted value


# The following function will calculate the P1 n-dimensional vector using the probability function
# to get the sigmoid probability function for the given lamda
def getP1(model, sample):
    tempP1 = np.matmul(sample, model)
    result = 1 / (1 + np.exp(tempP1))
    # result = np.exp(-temp) / (1 + np.exp(-temp))
    return result


# The following function will get the W matrix which is a diagonal matrix with the ith element
# being equal to Pr(y1=1|xi)*Pr(y1=0|xi)
def get_W(model_w, sample_w):
    x_b = np.matmul(sample_w, model_w)  # This is the expression inside the exponential (X)T*B
    pr1 = 1 / (1 + np.exp(x_b))  # Thi is Pr(yi=1|xi)
    pr0 = 1-pr1  # This is Pr(yi=0|xi)
    arr = pr1*pr0  # This is equals to Pr(y1=1|xi)*Pr(y1=0|xi)
    ans = np.diag(arr)  # this is the diagonal Matrix W
    return ans


for i in range(10):
    temp = label_train - getP1(beta, sample_train)  # This is equals to Y-P1 on the maximum # likelihood function
    L_derivative = -np.matmul(sample_train.transpose(), temp)  # This is the maximum likelihood function for beta

    temp_2 = -np.matmul(sample_train.transpose(), get_W(beta, sample_train))  # This is equals to (X)T*W
    hessian = np.matmul(temp_2, sample_train)  # This is the second derivative or (X)T*W*X
    if i != 0:
        beta = beta - np.matmul(L_derivative, np.linalg.inv(hessian))
        label_pred = (np.matmul(sample_test, beta) < 0) * 1

    err = sum(abs(label_test - label_pred)) / (n - num_train)
    err_matt.append(adj/err)
    beta_updates.append(i)

plt.plot(beta_updates[1:], err_matt[1:])
plt.xlabel('Beta Updates')
plt.ylabel('Testing Error')
plt.title('Testing Error vs Beta Updates')
plt.show()
