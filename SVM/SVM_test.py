import SVM_
import numpy as np


def data_genarate():
    x1 = np.random.uniform(1, 3, (100, 1))
    x2 = np.random.uniform(-3, -1, (100, 1))
    y1 = np.random.uniform(1.2, 3.2, (100, 1))
    y2 = np.random.uniform(-2.8, -0.7, (100, 1))
    data1 = np.hstack((x1, y1))
    data2 = np.hstack((x2, y1))
    data3 = np.hstack((x2, y2))
    data4 = np.hstack((x1, y2))
    label1 = np.ones((200, 1))
    label2 = -1 * np.ones((200, 1))
    data = np.vstack((data1, data3, data2, data4))
    label = np.vstack((label1, label2))
    return data, label


x, y = data_genarate()
# print(x, '\n', y)
classifier = SVM_.svm_train(x, y, 'gaussianKernel')
SVM_.result_plot(x, y, classifier, 'gaussianKernel')
