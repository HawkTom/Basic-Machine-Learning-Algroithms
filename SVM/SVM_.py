import numpy as np
import matplotlib.pyplot as plt


def svm_train(x, y, kernel="linear"):
    if kernel == "linear":
        K = np.dot(x, x.T)
    else:
        pass
    m = y.shape[0]
    alpha = np.zeros(y.shape)
    b = 0
    E = np.zeros(y.shape)
    eta, C = 0, 0
    L, H = 0, 0
    tol = 0.001
    for i in range(m):
        k_i = K[:, i]
        k_i.shape = y.shape
        E[i, 0] = b + sum(alpha * y * k_i) - y[i, 0]
        if ((y[i, 0] * E[i, 0] < -tol) & (alpha[i, 0] < C)) | ((y[i, 0] * E[i, 0] > tol) & (alpha[i, 0] > 0)):
            # select j, make sure i \neq j
            j = np.random.randint(m)
            while j == i:
                j = np.random.randint(m)
            # calculate error of j
            k_j = K[:, j]
            k_j.shape = y.shape
            E[j, 0] = b + sum(alpha * y * k_j) - y[j, 0]
            # save old alphas
            alpha_i_old, alpha_j_old = alpha[i, 0], alpha[j, 0]
            # calculate the L and H
            if y[i, 0] != y[j, 0]:
                L = max(0, alpha[j, 0] - alpha[i, 0])
                H = max(C, C + alpha[j, 0] - alpha[i, 0])
            else:
                L = max(0, alpha[j, 0] - alpha[i, 0] - C)
                H = max(C, alpha[j, 0] - alpha[i, 0])
            # compute eta
            eta = K[i, i] + K[j, j] - 2 * K[i, j]
            # update alpha_j
            alpha[j, 0] = alpha[j, 0] + y[i, 0] * (E[i, 0] - E[j, 0]) / eta
            # clip
            alpha[j, 0] = min(alpha[j, 0], H)
            alpha[j, 0] = max(alpha[j, 0], L)
            # update alpha_i
            alpha[i, 0] = alpha[i, 0] + y[i, 0] * \
                y[j, 0] * (alpha_j_old - alpha[j, 0])
            # update b
            b1 = b - E[i, 0] - y[i, 0] * (alpha[i, 0] - alpha_i_old) * \
                K[i, i] - y[j, 0] * (alpha[j, 0] - alpha_j_old) * K[i, j]
            b2 = b - E[j, 0] - y[i, 0] * (alpha[i, 0] - alpha_i_old) * \
                K[i, i] - y[j, 0] * (alpha[j, 0] - alpha_j_old) * K[j, j]
            if (alpha[i, 0] > 0) & (alpha[i, 0] < C):
                b = b1
            elif (alpha[j, 0] > 0) & (alpha[j, 0] < C):
                b = b2
            else:
                b = (b1 + b2) / 2
            break


def result_plot(x, y, classifier=None):
    place = np.argwhere(y == 1)
    label1 = x[place[:, 0]]
    place = np.argwhere(y == 0)
    label2 = x[place[:, 0]]
    plt.plot(label1[:, 0], label1[:, 1], '.', color='orange')
    plt.plot(label2[:, 0], label2[:, 1], '.', color='blue')
    plt.show()


with open("svm_data.txt", 'r') as f:
    datas = f.read()
    datas = datas.split('\n')
    x, y = [], []
    for sample in datas:
        s = sample.split('\t')
        x.append([float(s[0]), float(s[1])])
        y.append(float(s[2]))

x = np.array(x)
y = np.array(y)
y.shape = (y.shape[0], 1)
svm_train(x, y)
# print(x.shape, y.shape)
# result_plot(x, y)
