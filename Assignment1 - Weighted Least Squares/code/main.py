import numpy as np

# I borrowed code lines from the 4033_Code.py file from Canvas, only the ones to get and divide the data
# between 25% for testing and 75% for training. I also got inspired about how to get the error testing
# array, but the rest of the code is mine.

data = np.loadtxt('crime.csv', delimiter=',')

[n, p] = np.shape(data)
data = np.c_[np.ones(n), data]

num_train = int(0.75 * n)
num_test = int(0.25 * n)

max_iter = 10
# Crime Threshold
crime_th = 0.8
# weight for low crime rate
w_l = input("Enter weight for Low Crime Rates: ")
# weight for high crime rate
w_h = input("Enter weight for High Crime Rates: ")

count_l = 0
count_h = 0

# The following arrays will stored the test errors for all instances, the low crime rates and the low
# crime rates respectively
err_test_mat = []
err_test_mat_l = []
err_test_mat_h = []

# The following function will take the sample and label of the test sets, and it will return our model
# based on the equation derived for the weighted least squares
def beta(weight_temp, sample_train1, label_train1):
    t1 = np.matmul(sample_train1.transpose(), np.diag(weight_temp))
    temp1 = np.linalg.inv(np.matmul(t1, sample_train1))
    t2 = np.matmul(sample_train1.transpose(), np.diag(weight_temp))
    temp2 = np.matmul(t2, label_train1)
    beta_sol = np.matmul(temp1, temp2)
    return beta_sol

# The following function will take the sample and label tests, and it will return the error test
def error_calc(label, sample, beta1, num):
    label_pred = np.matmul(sample, beta1)
    err = np.linalg.norm(label - label_pred) / np.sqrt(num)
    return err


for i in range(max_iter):
    idx = np.random.permutation(n)
    idx_train = idx[0:num_train]
    idx_test = idx[n - num_test:]

    sample_train = data[idx_train, 0:-1]
    sample_test = data[idx_test, 0:-1]

    label_train = data[idx_train, -1]
    label_test = data[idx_test, -1]

    weight = data[idx_train, -1]  # This will be the diagonal weight matrix
    # The following indices are used to divide the high and low crime rate groups
    idx_l = []
    idx_h = []
    idx_test_l = []
    idx_test_h = []

    label_test_h = []
    sample_test_h = []

    # The following for loop is used to assign the weight values to the weigh diagonal matrix weight[]
    for j in range(num_train):
        if label_train[j] > crime_th:
            weight[j] = w_h
            idx_h.append(j)
        else:
            weight[j] = w_l
            idx_l.append(j)

    # The following for loop is used to separate the testing sample between low and high crime rates
    for j in range(num_test):
        if label_test[j] > crime_th:
            idx_test_h.append(j)
        else:
            idx_test_l.append(j)

    # Low crime rate set
    label_test_l = label_test[idx_test_l]
    sample_test_l = sample_test[idx_test_l, 0:]

    # High crime set
    label_test_h = label_test[idx_test_h]
    sample_test_h = sample_test[idx_test_h, 0:]

    # The following lines will take count of how many instances of low and high crime rates
    # are in the training set
    count_l += len(idx_l)
    count_h += len(idx_h)

    model = beta(weight, sample_train, label_train) # Model calculation (Beta)

    # The following lines will call  the function error_calc to get the testing error for the
    # whole data, the high crime rate and the low crime sets
    err_test = error_calc(label_test, sample_test, model, num_test)
    err_test_h = error_calc(label_test_h, sample_test_h, model, len(idx_test_h))
    err_test_l = error_calc(label_test_l, sample_test_l, model, len(idx_test_l))

    err_test_mat.append(err_test)
    err_test_mat_h.append(err_test_h)
    err_test_mat_l.append(err_test_l)

err_test = np.mean(err_test_mat)
std_test = np.std(err_test_mat)

err_test_h = np.mean(err_test_mat_h)
std_test_h = np.std(err_test_mat_h)

err_test_l = np.mean(err_test_mat_l)
std_test_l = np.std(err_test_mat_l)

avg_l = count_l / max_iter  # Number of training instances for the low crime set group
avg_h = count_h / max_iter  # Number of training instances for the high crime set group

print("Training Size for All Crime Rates= %d" % num_test)
print('Testing  RMSE for All Crime Rates= %.3f \u00B1 %.3f\n' % (err_test, std_test))

print("Training Size for High Crime Rates= %d" % len(idx_h))
print('Testing  RMSE for High Crime Rates= %.3f \u00B1 %.3f\n' % (err_test_h, std_test_h))

print("Training Size for Low Crime Rates= %d" % len(idx_l))
print('Testing  RMSE for Low Crime Rates= %.3f \u00B1 %.3f\n' % (err_test_l, std_test_l))

print("Average training instances for Low Crime Rates= %.3f" % avg_l)
print("Average training instances for High Crime Rates= %.3f" % avg_h)
