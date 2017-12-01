import numpy as np


def train(images, one_hot_lables):
    img = np.reshape(images, (len(images), 32 * 32 * 3))
    return [img, one_hot_lables]


def findmodes(vector):
    mode = {}
    for num in vector:
        if num not in mode:
            mode['num'] = 1
        else:
            mode['num'] += 1
    mode = sorted(mode.items(), key=lambda asd: asd[0], reverse=False)
    return int(mode[0][0])


def predict(images, model):
    K = 10
    data, train_lable = model[0], model[1]
    labels = np.zeros((len(images),), dtype=int)
    for i in range(len(images)):
        img = images[i]
        img = np.reshape(img, (1, 3072))
        disatance = np.sum(abs(data - img), axis=0)
        index = np.argsort(disatance)
        near_label = train_lable[0, index[0, :]]
        labels[i] = findmodes(near_label[0, 0:K])
        break
    return labels
