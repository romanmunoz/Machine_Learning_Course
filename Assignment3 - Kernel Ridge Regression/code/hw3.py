import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib import pyplot as plt

# I borrow the read data and the split data algorithm from the Canvas code provided, but I
# left notes in all my implementations
data = np.loadtxt('crime.csv', delimiter=',')

[n, p] = np.shape(data)

num_train = int(0.75 * n)
num_test = int(0.25 * n)

max_iter = 5

# These are the arbitrary values of m, the last value is equal to n, or num_train = 1494
m = [10, 50, 100, 150, 200, 250]
alpha = 0  # This is the initial value of alpha
gamma = 0.001  # This gamma value will be used in the rbf kernel function
lam = 0.1  # This lambda will be used to determine the optimal model based on the optimization calculated

err_avg_mat = []  # This is the error average matriz

err_sum = [np.zeros(len(m))]  # This matrix adds all the instances of each m value


for i in range(max_iter):
    err_test_mat = []  # This is the test error use for each iteration
    idx = np.random.permutation(n)
    idx_train = idx[0:num_train]
    idx_test = idx[n - num_test:]

    sample_train = data[idx_train, 0:-1]
    sample_test = data[idx_test, 0:-1]

    label_train = data[idx_train, -1]
    label_test = data[idx_test, -1]

    for j in range(len(m)):
        x1 = sample_train[0:m[j], :]  # x1 is the new sample which is reduced according to m
        K1 = rbf_kernel(sample_train, x1, gamma=gamma)  # This is the first Kernel function which is nxm
        K2 = rbf_kernel(x1, x1, gamma=gamma)  # This second Kernel function is mxm

        temp1 = np.matmul(K1.transpose(), K1)  # This is the matrix multiplication for the first Kernel Function
        temp2 = lam*K2  # this multiplies lamda times the second Kernel function which is mxm
        temp3 = np.linalg.inv(temp1 + temp2)  # Inverse of the first two operands
        temp4 = np.matmul(K1.transpose(), label_train)  # These are the arguments outside the inverse functions
        alpha = np.matmul(temp3, temp4)  # Final result for the optimal model stored as alpha

        KT = rbf_kernel(sample_test, x1, gamma=gamma)  # This new Kernel function is used for testing

        # The following lines perform the RMSE calculation of each instance of m
        label_test_pred = np.matmul(KT, alpha)
        err = np.linalg.norm(label_test - label_test_pred) / np.sqrt(num_test)
        err_test = np.sqrt(mean_squared_error(label_test, label_test_pred))
        err_test_mat.append(err_test)
    err_sum = err_sum + err_test
err_avg_mat = np.divide(err_sum, len(m))  # Average of each m instance inside the err_sum matrix
err_avg_mat = np.reshape(err_avg_mat, (6,)).transpose()

#  The following lines will plot the RMSE vs m
plt.plot(m, err_test_mat)
plt.xlabel('Instances of m')
plt.ylabel('Testing Error')
plt.title('RMSE vs m')
plt.show()

