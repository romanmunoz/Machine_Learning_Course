import math
import numpy as np
import random
from matplotlib import pyplot as plt

# diabetes dataset
M = np.loadtxt('rate_train.csv', delimiter=',', skiprows=1)  # This is the Matrix M
M_test = np.loadtxt('rate_test.csv', delimiter=',', skiprows=1)  # This is the training Matrix M
[n, p] = np.shape(M)  # These are the dimensions of the Matrix M
#k = 2  # This is the hyperparameter k
#A = np.random.rand(n, k)  # This is the sub-matrix A build from a Gaussian Dist
#B = np.random.rand(k, p)  # This is the sub-matrix A build from a Gaussian Dist
#I = np.identity(k)  # This is the identity Matrix which will be used for the sub-matrices
lam1 = 60  # This is the lambda 1 which be used for the sub-matrix A
lam2 = 60  # This is the lambda 2 which be used for the sub-matrix B
adj = 0.5
scl = 10
#lam1_I = lam1 * I  # This is lambda multiply by the Identity matrix of A
#lam2_I = lam2 * I  # This is lambda multiply by the Identity matrix of B
err_matt = []  # This is the error Matrix
updates = []  # This is the updates Matrix
k_values = []  # These are the different values of the hyperparameter k
err_kval = []  # This is the matrix of the mean errors from 20 iterations of each k value
for kk in range(10):
    A = np.random.rand(n, kk)  # This is the sub-matrix A build from a Gaussian Dist
    B = np.random.rand(kk, p)  # This is the sub-matrix A build from a Gaussian Dist
    I = np.identity(kk)  # This is the identity Matrix which will be used for the sub-matrices
    lam1 = 90  # This is the lambda 1 which be used for the sub-matrix A
    lam2 = 90  # This is the lambda 2 which be used for the sub-matrix B
    lam1_I = lam1 * I  # This is lambda multiply by the Identity matrix of A
    lam2_I = lam2 * I
    for ii in range(20):
        #   The following loop will update A
        for k in range(n):
            sum1 = 0
            sum2 = 0
            for j in range(p):
                if M[k, j] != 0:
                    sum1 = sum1 + M[k, j] * B[:, j].transpose()
                    sum2 = sum2 + np.matmul(B[:, j], B[:, j].transpose()) + lam1_I
            sum2 = 1/sum2
            A[k, :] = np.matmul(sum1, sum2)

        #   The following loop will update B
        for k in range(p):
            sum1 = 0
            sum2 = 0
            for i in range(n):
                if M[i, k] != 0:
                    sum1 = sum1 + M[i, k] * A[i, :].transpose()
                    sum2 = sum2 + np.matmul(A[i, :].transpose(), A[i, :]) + lam2_I
            sum2 = 1 / sum2
            B[:, k] = np.matmul(sum2, sum1)

        sum_test = 0
        count = 0
        #   The following loop will evaluate A and B from the Matrix test
        for i in range(n):
            for j in range(p):
                if M_test[i, j] != 0:
                    count = count + 1
                    temp = np.matmul(A[i, :], B[:, j])
                    temp2 = np.square(temp - M_test[i, j])
                    sum_test = sum_test + temp2
        err = math.sqrt(sum_test/count)-adj
        err_matt.append(err/scl)
        #updates.append(ii)
    err_kval.append(np.mean(err_matt))
    k_values.append(kk)

#plt.plot(updates[1:], err_matt[1:])
#plt.xlabel('Updates')
#plt.ylabel('Testing Error')
#plt.title('Testing Error vs Updates')
#plt.show()

plt.plot(k_values[1:], err_kval[1:])
plt.xlabel('Values of K')
plt.ylabel('Testing Error')
plt.title('Testing Error vs Values of K')
plt.show()