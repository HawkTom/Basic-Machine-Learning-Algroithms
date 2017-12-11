import pandas as pd
import numpy as np
from collections import Counter


def read_file(FileName):
    data = pd.read_csv(FileName)
    data = data.as_matrix()
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    train_data = data[0:int(data.shape[0] * 0.9), :]
    test_data = data[0:int(data.shape[0] * 0.1), :]
    return train_data, test_data


def train(data):
    token = list(set(' '.join(data[:, 0]).split()))
    data1 = data[data[:, 1] == '1'][:, 0]
    data0 = data[data[:, 1] == '0'][:, 0]

    spam, ham = [], []
    for d in data1:
        spam.append(set(d.split()))
    for d in data0:
        ham.append(set(d.split()))
    spam_token = np.zeros((1, len(token)))
    ham_token = np.zeros((1, len(token)))

    for i in range(len(token)):
        for email in spam:
            if token[i] in email:
                spam_token[0, i] += 1
        spam_token[0, i] /= len(spam)
        for email in ham:
            if token[i] in email:
                ham_token[0, i] += 1
        ham_token[0, i] /= len(ham)
    spam_p = dict(zip(token, spam_token[0, :]))
    ham_p = dict(zip(token, ham_token[0, :]))
    return [spam_p, ham_p]


def predict(model, data):
    spam_p, ham_p = model[0], model[1]
    # data1 = data[data[:, 1] == '1'][:, 0]
    # data0 = data[data[:, 1] == '0'][:, 0]
    accuracy = np.zeros((1, data.shape[0]))
    for j in range(data.shape[0]):
        token = list(set(data[j, 0]))
        p_s_w = np.zeros((1, len(token)))
        for i in range(len(token)):
            p_w_s, p_w_h = 0, 0
            flag_s = token[i] in spam_p
            flag_h = token[i] in ham_p
            if flag_s & flag_h:
                p_w_s = spam_p[token[i]]
                p_w_h = ham_p[token[i]]
            elif flag_s & (not flag_h):
                p_w_s = spam_p[token[i]]
                p_w_h = 0.01
            elif not flag_s & flag_h:
                p_w_s = 0.01
                p_w_h = ham_p[token[i]]
            else:
                p_s_w[0, i] = 0.4
                continue
            p_s_w[0, i] = p_w_s / (p_w_h + p_w_s)
        p = np.prod(p_s_w) / (np.prod(p_s_w) + np.prod(1 - p_s_w))
        if p > 0.9:
            label = 1
        else:
            label = 0
        if data[j, 1] == label:
            accuracy[0, j] = 1
        else:
            accuracy[0, i] = 0
    return np.sum(accuracy, axis=1) / accuracy.shape[1]


def main():
    train_data, test_data = read_file(r'assignment1_data.csv')
    model = train(train_data)
    accuracy = predict(model, test_data)
    print(accuracy)


if __name__ == '__main__':
    main()
