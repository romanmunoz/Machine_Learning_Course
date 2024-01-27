import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib import pyplot as plt

data = np.loadtxt('crime.csv', delimiter=',')

[n, p] = np.shape(data)

num_train = int(0.75 * n)
num_test = int(0.25 * n)

max_iter = 10

err_avg_mat = []
err_test_mat = []
err_sum = []

# These are the arbitrary values of m, the last value is equal to n, or num_train = 1494
m = [10, 50, 100, 200, 400, num_test]
alpha = 0  # This is the initial value of alpha
gamma = 0.001
lam = 0.1

for i in range(1):
    idx = np.random.permutation(n)
    idx_train = idx[0:num_train]
    idx_test = idx[n - num_test:]

    sample_train = data[idx_train, 0:-1]
    sample_test = data[idx_test, 0:-1]

    label_train = data[idx_train, -1]
    label_test = data[idx_test, -1]

    for j in range(6):
        x1 = sample_train[0:m[j], :]
        K1 = rbf_kernel(sample_train, x1, gamma=gamma)
        K2 = rbf_kernel(x1, x1, gamma=gamma)

        temp1 = np.matmul(K1.transpose(), K1)
        temp2 = lam*K2
        temp3 = np.linalg.inv(temp1 + temp2)
        temp4 = np.matmul(K1.transpose(), label_train)
        alpha = np.matmul(temp3, temp4)

        KT = rbf_kernel(sample_test, x1, gamma=gamma)

        label_test_pred = np.matmul(KT, alpha)
        err = np.linalg.norm(label_test - label_test_pred) / np.sqrt(num_test)
        err_test = np.sqrt(mean_squared_error(label_test, label_test_pred))
        err_test_mat.append(err_test)
    err_sum = err_sum + err_test_mat

err_avg_mat = np.divide(err_sum, max_iter)

#  The following lines will plot the RMSE vs m
plt.plot(m, err_avg_mat)
plt.xlabel('Instances of m')
plt.ylabel('Testing Error')
plt.title('RMSE vs m')
plt.show()
