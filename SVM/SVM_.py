import numpy as np
import matplotlib.pyplot as plt


def svm_train(x, y, kernel="linear"):
    if kernel == "linear":
        K = np.dot(x, x.T)
    else:
        pass
    place = np.argwhere(y == 0)
    y[place[:, 0]] = -1
    m = y.shape[0]
    alpha = np.zeros(y.shape)
    b = 0
    E = np.zeros(y.shape)
    eta, C = 0, 1
    L, H = 0, 0
    tol, passes = 0.001, 0
    while passes < 20:
        num_changed_alphas = 0
        for i in range(m):
            k_i = K[:, i]
            k_i.shape = y.shape
            E[i, 0] = b + sum(alpha * y * k_i) - y[i, 0]
            if ((y[i, 0] * E[i, 0] < -tol) & (alpha[i, 0] < C)) \
                    | ((y[i, 0] * E[i, 0] > tol) & (alpha[i, 0] > 0)):
                # select j, make sure i \neq j
                j = np.random.randint(m)
                while j == i:
                    j = np.random.randint(m)
                # j = 21
                # calculate error of j
                k_j = K[:, j]
                k_j.shape = y.shape
                E[j, 0] = b + sum(alpha * y * k_j) - y[j, 0]
                # save old alphas
                alpha_i_old, alpha_j_old = alpha[i, 0], alpha[j, 0]
                # calculate the L and H
                if y[i, 0] == y[j, 0]:
                    L = max(0, alpha[j, 0] + alpha[i, 0] - C)
                    H = min(C, alpha[j, 0] + alpha[i, 0])
                else:
                    L = max(0, alpha[j, 0] - alpha[i, 0])
                    H = min(C, C + alpha[j, 0] - alpha[i, 0])
                if L == H:
                    continue
                # compute eta
                eta = - K[i, i] - K[j, j] + 2 * K[i, j]
                if eta >= 0:
                    continue
                # update alpha_j
                alpha[j, 0] = alpha[j, 0] - y[j, 0] * (E[i, 0] - E[j, 0]) / eta
                # clip
                alpha[j, 0] = min(alpha[j, 0], H)
                alpha[j, 0] = max(alpha[j, 0], L)
                if abs(alpha[j, 0] - alpha_j_old) < tol:
                    alpha[j, 0] = alpha_j_old
                    continue
                # update alpha_i
                alpha[i, 0] = alpha[i, 0] + y[i, 0] * \
                    y[j, 0] * (alpha_j_old - alpha[j, 0])
                # update b
                b1 = b - E[i, 0] - y[i, 0] * (alpha[i, 0] - alpha_i_old) * \
                    K[i, j] - y[j, 0] * (alpha[j, 0] - alpha_j_old) * K[i, j]
                b2 = b - E[j, 0] - y[i, 0] * (alpha[i, 0] - alpha_i_old) * \
                    K[i, j] - y[j, 0] * (alpha[j, 0] - alpha_j_old) * K[j, j]
                if (alpha[i, 0] > 0) & (alpha[i, 0] < C):
                    b = b1
                elif (alpha[j, 0] > 0) & (alpha[j, 0] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    w = np.dot((alpha * y).T, x)
    return [w, b]


def result_plot(x, y, classifier=None):
    place = np.argwhere(y == 1)
    label1 = x[place[:, 0]]
    place = np.argwhere(y == -1)
    label2 = x[place[:, 0]]
    plt.plot(label1[:, 0], label1[:, 1], '.', color='orange')
    plt.plot(label2[:, 0], label2[:, 1], '.', color='blue')

    if classifier:
        x = np.linspace(min(x[:, 0]), max(x[:, 1]), 100)
        y = -(classifier[0][0][0] * x + classifier[1]) / classifier[0][0][1]
        plt.plot(x, y)

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
classifier = svm_train(x, y)
# classifier = [np.array([[1, 2]]), 1]
print(classifier)
result_plot(x, y, classifier)
