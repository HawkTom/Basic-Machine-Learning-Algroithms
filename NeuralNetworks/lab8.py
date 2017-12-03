import numpy as np


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
    if (derivative == False):
        return x * (x > 0)
    else:
        return 1 * (x > 0)


def Softmax(x):
    sfm = np.exp(x) / sum(np.exp(x))
    return sfm


def createNN(inputSize, outputSize, numberLayers):
    hiddenSize = 60
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


def BP(Network, actual_label, predict_label, learning_rate=0.8):
    inputLayer, hiddenLayer, outputLayer = Network[0], Network[1], Network[2]

    alpha = learning_rate

    delta_softmax = predict_label - actual_label

    delta_output = delta_softmax
    delta_hidden = np.dot(outputLayer.weights.T, delta_output) * \
        derivative_sigmoid(hiddenLayer.input)

    gradient1 = np.sum(delta_output, axis=1) / actual_label.shape[1]
    gradient1.shape = outputLayer.bias.shape
    outputLayer.bias = outputLayer.bias - alpha * gradient1

    gradient2 = np.dot(delta_output, hiddenLayer.output.T) / \
        actual_label.shape[1]
    outputLayer.weights = outputLayer.weights - alpha * gradient2

    gradient3 = np.sum(alpha * delta_hidden, axis=1) / actual_label.shape[1]
    gradient3.shape = hiddenLayer.bias.shape
    hiddenLayer.bias = hiddenLayer.bias - alpha * gradient3

    gradient4 = np.dot(delta_hidden, inputLayer.output.T) / \
        actual_label.shape[1]
    hiddenLayer.weights = hiddenLayer.weights - alpha * gradient4
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def train(images, one_hot_labels):
    NN = createNN(1024, 10, 3)
    for i in range(50000):
        img = images[i]
        img = np.reshape(img, (1, 32 * 32 * 3))
        actual_label = one_hot_labels[i]
        actual_label.shape = (10, 1)
        actual_label.T
        predict_label = forward(NN, img)
        print(actual_label.shape, predict_label.shape)
        error = np.sqrt(sum((predict_label - actual_label)**2))
        print(error)
        NN = BP(NN, actual_label, predict_label)
        break
    return NN


def train_Batch(images, one_hot_labels):
    NN = createNN(3072, 10, 3)
    batch_number = 100
    batch_size = int(len(images) / batch_number)
    for j in range(20):
        for i in range(batch_number):
            img = images[batch_size * i:batch_size * (i + 1)]
            # img = np.mean(np.reshape(img, (batch_size, 32 * 32, 3)), axis=2)
            img = np.reshape(img, (batch_size, 32 * 32 * 3))  # img: n*3072
            actual_label = one_hot_labels[batch_size *
                                          i:batch_size * (i + 1), :]
            actual_label = actual_label.T  # label: 10*n
            predict_label = forward(NN, img)
            error = -sum(sum(actual_label * np.log(predict_label))
                         ) / batch_size
            print(j, i, error)
            NN = BP(NN, actual_label, predict_label)
    return NN


def predict(images, NN):
    print(images.shape)
    labels = np.zeros((len(images),), dtype=int)
    for i in range(len(images)):
        img = images[i]
        img = np.reshape(img, (1, 32 * 32 * 3))
        predict_label = forward(NN, img)
        # print(predict_label)
        labels[i] = np.argmax(predict_label)
    # Return a Numpy ndarray with the length of len(images).
    # e.g. np.zeros((len(images),), dtype=int) means all predictions are 'airplane's
    print(labels)
    return labels
