import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt


def svm_train(x, y, kernel="linear"):
    if kernel == "linear":
        K = np.dot(x, x.T)
    elif kernel == "gaussianKernel":
        sigma = 0.1
        K = dist.cdist(x, x)
        K = np.exp(-K**2 / (2 * sigma * sigma))
    elif kernel == "polynomialKernel":
        d = 2
        K = np.dot(x, x.T)
        K = K**d
    m = y.shape[0]
    alpha = np.zeros(y.shape)
    b = 0
    E = np.zeros(y.shape)
    eta, C = 0, 1
    L, H = 0, 0
    tol, passes = 0.001, 0
    while passes < 5:
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
    return [w, b, alpha]


def result_plot(x, y, classifier=None, kernel='linear'):
    place = np.argwhere(y == 1)
    label1 = x[place[:, 0]]
    place = np.argwhere(y == -1)
    label2 = x[place[:, 0]]
    plt.plot(label1[:, 0], label1[:, 1], '.', color='orange')
    plt.plot(label2[:, 0], label2[:, 1], '.', color='blue')

    if len(classifier) > 1:
        if kernel == 'linear':
            x = np.linspace(min(x[:, 0]), max(x[:, 1]), 100)
            y = -(classifier[0][0][0] * x + classifier[1]) / \
                classifier[0][0][1]
            plt.plot(x, y)
        else:
            x1 = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
            x2 = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
            X, Y = np.meshgrid(x1, x2)
            vals = np.zeros_like(X)
            for i in range(X.shape[1]):
                this_X = np.vstack((X[:, i], Y[:, i]))
                this_X = this_X.T
                vals[:, i] = svmPredict(x, y, classifier, this_X, kernel)
            plt.contour(X, Y, vals)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def svmPredict(x, y, model, X, kernel):
    """
    x: train data
    y: train label
    X: test data
    model[2]: alpha
    """
    print(x.shape, X.shape)
    alpha = model[2]  # alpha
    if kernel == "gaussianKernel":
        sigma = 0.1
        K = dist.cdist(x, X)
        K = np.exp(-K**2 / (2 * sigma * sigma))
    elif kernel == "polynomialKernel":
        pass
    else:
        pass
    K = y * K
    K = alpha * K
    p = np.sum(K, axis=0) + model[1]
    p.shape = (1, X.shape[0])
    place = np.argwhere(p >= 0)
    p[place[:, 0]] = 1
    place = np.argwhere(p < 0)
    p[place[:, 0]] = 0
    return p


def main():
    with open("svm_data2.txt", 'r') as f:
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
    place = np.argwhere(y == 0)
    y[place[:, 0]] = -1
    print(x.shape, y.shape)
    classifier = svm_train(x, y, 'gaussianKernel')
    # np.save('SVM', classifier)
    # classifier = np.load('SVM.npy')
    # print(classifier)
    result_plot(x, y, classifier, 'gaussianKernel')


if __name__ == '__main__':
    main()
