import pandas as pd
import numpy as np
import re
# import time


def read_file(FileName):
    data = pd.read_csv(FileName)
    data = data.as_matrix()
    for i in range(data.shape[0]):
        data[i, 0] = textParaser(data[i, 0])
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    train_data = data[0:int(data.shape[0] * 0.9), :]
    test_data = data[0:int(data.shape[0] * 0.1), :]
    return train_data, test_data

def textParaser(text):
    p = re.compile(r'[^a-zA-Z]|\d')
    words = p.split(text)
    words = [word.lower() for word in words if len(word) > 2]
    return words

def train(data):
    token = list(set(np.hstack(data[:, 0])))
    spam = data[data[:, 1] == 1][:, 0]
    ham = data[data[:, 1] == 0][:, 0]

    spam_token = np.zeros((len(spam), len(token)))
    ham_token = np.zeros((len(ham), len(token)))

    for i in range(len(spam)):
        for word in spam[i]:
            if word in token:
                spam_token[i, token.index(word)] += 1
    for i in range(len(ham)):
        for word in ham[i]:
            if word in token:
                ham_token[i, token.index(word)] += 1
    # spam_token = np.load('spam_token.npy')
    # ham_token = np.load('ham_token.npy')

    spam_token = np.sum(spam_token, axis=0) / len(spam)
    ham_token = np.sum(ham_token, axis=0) / len(ham)

    spam_p = dict(zip(token, spam_token))
    ham_p = dict(zip(token, ham_token))

    return [spam_p, ham_p]


def predict(model, data):
    spam_p, ham_p = model[0], model[1]

    accuracy = np.zeros((1, data.shape[0]))
    for j in range(data.shape[0]):
        words = data[j, 0]
        p_s_w = np.zeros((1, len(words)))
        for i in range(len(words)):
            if words[i] not in spam_p:
                p_s_w[0, i] = 0.4
                continue
            p_w_s, p_w_h = 0, 0
            p_w_s = spam_p[words[i]]
            p_w_h = ham_p[words[i]]
            if p_w_s == 0.0:
                p_w_s = 0.01
            if p_w_h == 0.0:
                p_w_h = 0.01
            p_s_w[0, i] = p_w_s / (p_w_h + p_w_s)
        p = np.prod(p_s_w) / (np.prod(p_s_w) + np.prod(1 - p_s_w))
        if p > 0.8:
            label = 1
        else:
            label = 0
        if data[j, 1] == label:
            accuracy[0, j] = 1
        else:
            accuracy[0, j] = 0

    return np.sum(accuracy, axis=1) / accuracy.shape[1]


def main():
    print("Reading Data .........")
    train_data, test_data = read_file(r'assignment1_data.csv')
    print("Finish Read ! ")
    print("Training .......")
    model = train(train_data)
    print("Finsih Train !")
    print("Predincting .......")
    accuracy = predict(model, test_data)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()
