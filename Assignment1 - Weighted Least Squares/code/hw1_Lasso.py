import numpy as np
import random
from matplotlib import pyplot as plt

# I borrowed code lines from the 4033_Code.py file from Canvas, only the ones to get and divide the data
# between 25% for testing and 75% for training. I also got inspired about how to get the error testing
# array, but the rest of the code is mine.

data = np.loadtxt('crime.csv', delimiter=',')

[n, p] = np.shape(data)
data = np.c_[np.ones(n), data]

num_train = int(0.75 * n)
num_test = int(0.25 * n)

i = np.random.permutation(n)
idx_train = i[0:num_train]
idx_test = i[n - num_test:]

sample_train = data[idx_train, 0:-1]
sample_test = data[idx_test, 0:-1]

label_train = data[idx_train, -1]
label_test = data[idx_test, -1]

err_train_mat = []
err_test_mat = []

beta = np.random.normal(0, 0.01, size=(p, 1))
beta_0 = np.copy(beta)  # A copy of the beta array by value
beta_0[0] = 0  # Beta[0] assign to zero
lam = 0.1  # lambda
max_updates = 400  # Number of Coordinate Descent updates
nonzero = 0  # Number of non-zero elements on Beta array

# The following arrays will be used to graph the Testing Error vs the CD Updates
cd_updates = []
graph_test = []
graph_train = []

graph_nonzero = []  # This array will contain the number of nonzero elements in the model (Beta)


# This Function will return the error array from the prediction model
def error_calc(err_matt, label, sample, model, num):
    label_pred = np.matmul(sample, model)
    err = np.linalg.norm(label - label_pred) / np.sqrt(num)
    err_matt.append(err)
    return err_matt


# The following function will get the value of data based on the parameters from the sample, label and
# t values. The math is done on Matrix Form
def get_Delta(beta_temp, sample, label, t):
    beta_new = np.copy(beta_temp)
    beta_new[t] = 0
    label = np.reshape(label, (1494, 1))
    temp1 = np.matmul(sample, beta_new) - label
    temp2 = np.matmul(sample[:, t].transpose(), temp1)
    d = -2 * temp2
    return d


for i in range(max_updates):
    idx = random.randint(0, p - 1)
    if idx == 0:  # Case 1: Beta(0)
        temp_2 = 0
        for j in range(0, p):
            temp = np.matmul(sample_train[j].transpose(), beta_0) - label_train[j]
            temp_2 = temp_2 + temp
        beta[idx] = (-1 / p) * temp_2
    elif beta[idx] > 0:  # Case 2: Beta[t] > 0
        if get_Delta(beta, sample_train, label_train, idx) > lam:  # Check for Delta > lamda
            temp = get_Delta(beta, sample_train, label_train, idx) - lam
            beta[idx] = temp / (2 * (np.matmul(sample_train[:, idx].transpose(), sample_train[:, idx])))
        else:
            beta[idx] = 0
    elif beta[idx] < 0:  # Case 3: Beta[t] < 0
        if get_Delta(beta, sample_train, label_train, idx) < -lam:  # Check for Delta < -lamda
            temp = get_Delta(beta, sample_train, label_train, idx) + lam
            beta[idx] = temp / (2 * (np.matmul(sample_train[:, idx].transpose(), sample_train[:, idx])))
        else:
            beta[idx] = 0
    else:  # For all other cases, Beta[t] = 0, this line gets rid of some features
        beta[idx] = 0

    # Reshape the size of beta for better calculations
    beta = np.reshape(beta, (101,))

    # The following lines will call the error_calc to get the testing error for each group
    err_train_mat = error_calc(err_train_mat, label_train, sample_train, beta, num_train)
    err_test_mat = error_calc(err_test_mat, label_test, sample_test, beta, num_test)

    err_train = np.mean(err_train_mat)
    err_test = np.mean(err_test_mat)

    beta = np.reshape(beta, (101, 1))  # Reshape Beta to its original form

    # The following arrays will be used to graph the Testing Error vs the CD Updates
    cd_updates.append(i)
    graph_test.append(err_test)

    # The following array graph_nonzero will contain the number of non-zero elements in the model
    nonzero = np.count_nonzero(beta)
    graph_nonzero.append(nonzero)

#  The following lines will plot the Testing Error vs the CD Updates
plt.plot(cd_updates, graph_test)
plt.xlabel('CD Updates')
plt.ylabel('Testing Error')
plt.title('Testing Error vs CD Updates')
plt.show()

#  The following lines will plot the non-zero elements in the array vs the CD Updates
plt.plot(cd_updates, graph_nonzero)
plt.xlabel('CD Updates')
plt.ylabel('Number of non-zero elements')
plt.title('Non-zero elements vs CD Updates')
plt.show()
