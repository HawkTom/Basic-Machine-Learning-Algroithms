import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


class layer(object):
    def __init__(self, numberNodes_ThisLayer):
        self.numberNodes = numberNodes_ThisLayer
        self.input = np.zeros((numberNodes_ThisLayer, 1))

    def activate(self, fun='sigmoid'):
        if fun == 'sigmoid':
            self.output = Sigmoid(self.input)
        elif fun == 'softmax':
            self.output = Softmax(self.input)

    def createParameters(self, numberNods_LastLayer):
        self.weights = np.random.randn(self.numberNodes, numberNods_LastLayer)
        self.bias = np.random.randn(self.numberNodes, 1)


def Sigmoid(x):
    sm = 1 / (np.exp(-x) + 1)
    return sm


def derivative_sigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))


def ReLu(x, derivative=False):
    if ~derivative:
        return x * (x > 0)
    else:
        return 1 * (x > 0)


def Softmax(x):
    sfm = np.exp(x) / sum(np.exp(x))
    return sfm


def createNN(inputSize, outputSize, numberLayers):
    hiddenSize = 30
    inputLayer = layer(inputSize)
    hiddenLayer = layer(hiddenSize)
    hiddenLayer.createParameters(inputSize)  # weights: 35x784   bias: 35x1
    outputLayer = layer(outputSize)
    outputLayer.createParameters(hiddenSize)  # weights: 10x35    bias:10x1
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def forward(Network, data):
    inputLayer = Network[0]
    inputLayer.output = data.T
    # inputLayer.activate()

    hiddenLayer = Network[1]
    hiddenLayer.input = np.dot(
        hiddenLayer.weights, inputLayer.output) + hiddenLayer.bias
    hiddenLayer.activate()
    # print(hiddenLayer.output)

    outputLayer = Network[2]
    outputLayer.input = np.dot(
        outputLayer.weights, hiddenLayer.output) + outputLayer.bias
    outputLayer.activate(fun='softmax')

    return outputLayer.output


def weight_cal(x, y):
    out = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[1]):
        a = x[:, i]
        a.shape = (x.shape[0], 1)
        b = y[i, :]
        b.shape = (1, y.shape[1])
        out = out + np.dot(a, b)
    out = out / x.shape[1]
    return out


def BP(Network, actual_label, predict_label, learning_rate=1):
    inputLayer, hiddenLayer, outputLayer = Network[0], Network[1], Network[2]

    alpha = learning_rate

    delta_softmax = predict_label - actual_label

    delta_output = delta_softmax
    delta_hidden = np.dot(outputLayer.weights.T, delta_output) * \
        derivative_sigmoid(hiddenLayer.input)

    gradient1 = np.sum(delta_output, axis=1) / actual_label.shape[1]
    gradient1.shape = outputLayer.bias.shape
    outputLayer.bias = outputLayer.bias - alpha * gradient1

    gradient2 = weight_cal(delta_output, hiddenLayer.output.T)
    outputLayer.weights = outputLayer.weights - alpha * gradient2

    gradient3 = np.sum(alpha * delta_hidden, axis=1) / actual_label.shape[1]
    gradient3.shape = hiddenLayer.bias.shape
    hiddenLayer.bias = hiddenLayer.bias - alpha * gradient3

    gradient4 = weight_cal(delta_hidden, inputLayer.output.T)
    hiddenLayer.weights = hiddenLayer.weights - alpha * gradient4
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def trainNN(Network, datas):
    for j in range(2):
        for i in range(datas.shape[0]):  # datas.shape[0]
            data = datas[i:i + 1]
            label_index = data['label']
            pixel_data = data.drop(['label'], axis=1).values / 255
            actual_label = np.zeros((Network[-1].numberNodes, 1))
            actual_label[label_index.values[0], 0] = 1
            predict_label = forward(Network, pixel_data)
            error = np.sqrt(sum((predict_label - actual_label)**2))
            print(error)
            Network = BP(Network, actual_label, predict_label)
    return Network


def trainNN_BatchGD(Network, datas):
    batch_size = 100
    batch_number = int(datas.shape[0] / batch_size)
    for j in range(10):
        #        print(batch_size, batch_number)
        for i in range(batch_number):  # batch_number
            data = datas[batch_size * i:batch_size * (i + 1)]
            label_index = data['label'].values
            # pixel data: 42000*784
            pixel_data = data.drop(['label'], axis=1).values / 255
            actual_label = np.zeros((Network[-1].numberNodes, data.shape[0]))
            for i in range(data.shape[0]):
                actual_label[label_index[i], i] = 1   # actual label: 10x42000
            # predict label: 10x42000
            predict_label = forward(Network, pixel_data)
            Network = BP(Network, actual_label, predict_label)
            error = -sum(sum(actual_label * np.log(predict_label))
                         ) / batch_size
            print(error)
    return Network


def testNN(Network, testdata):
    # x = [151, 152, 153, 154, 155]
    result = []
    for i in range(testdata.shape[0]):  # testdata.shape[0]
        pixel_data = testdata[i:i + 1].values
        predict_label = forward(Network, pixel_data)
        predict_num = np.argmax(predict_label)
        result.append(str(i + 1) + ',' + str(predict_num))
#        plt.subplot(x[i-55])
#        pic = np.reshape(pixel_data, (28,28))
#        plt.imshow(pic)
    return result


def visiualization(datas):
    x = [151, 152, 153, 154, 155]
    for i in range(25, 30):
        plt.subplot(x[i - 25])
        data = datas[i:i + 1].values
        pic = np.reshape(data, (28, 28))
        # data = datas[i:i+1]
        # pixel_data = data.drop(['label'], axis=1).values
        # pic = np.reshape(pixel_data, (28,28))
        plt.imshow(pic)


def main():
    start = time.time()
    trainFile = "train.csv"
    data = pd.read_csv(trainFile)
    NN = createNN(784, 10, 3)
    NN = trainNN_BatchGD(NN, data)
    #
    #
    #NN = createNN(784, 10, 3)
    #NN = trainNN(NN, data)
    # np.save('NN', NN)

    # testFile = "test.csv"
    # testdata = pd.read_csv(testFile)
    #NN = np.load('NN.npy')
    # result_ = testNN(NN, testdata)
    # print(result_)
    # with open("result.csv",'w') as f:
    #    f.write('ImageId,Label\n'+'\n'.join(result_))

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
    print("OK")
