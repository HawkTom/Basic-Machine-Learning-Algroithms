import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


class layer(object):
    def __init__(self, numberNodes_ThisLayer):
        self.numberNodes = numberNodes_ThisLayer
        self.input = np.zeros((numberNodes_ThisLayer, 1))

    def activate(self):
        self.output = Softmax(self.input)

    def createParameters(self, numberNods_LastLayer):
        self.weights = np.random.randn(self.numberNodes, numberNods_LastLayer)
        self.bias = np.random.randn(self.numberNodes, 1)


def Sigmoid(x):
    sm = 1 / (np.exp(-x) + 1)
    return sm

def derivative_sigmoid(x):
    return Sigmoid(x)*(1- Sigmoid(x))

def Softmax(x):
    sfm = np.exp(x)/sum(np.exp(x))
    return sfm

def derivative_softmax(x):
    return Softmax(x)*(1-Softmax(x))

def createNN(inputSize, outputSize, numberLayers):
    hiddenSize = 35
    inputLayer = layer(inputSize)
    hiddenLayer = layer(hiddenSize)
    hiddenLayer.createParameters(inputSize)
    outputLayer = layer(outputSize)
    outputLayer.createParameters(hiddenSize)
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def predict(Network, data):
    inputLayer = Network[0]
    inputLayer.input = data.T
    inputLayer.activate()
    
    hiddenLayer = Network[1]
    hiddenLayer.input = np.dot(hiddenLayer.weights, inputLayer.output) + hiddenLayer.bias
    hiddenLayer.activate()
    
    outputLayer = Network[2]
    outputLayer.input = np.dot(outputLayer.weights, hiddenLayer.output) + outputLayer.bias
    outputLayer.activate()    
    return outputLayer.output


def BP(Network, actual_label, predict_label, learning_rate = 0.2):
    inputLayer, hiddenLayer, outputLayer = Network[0], Network[1], Network[2]
    
    delta_output = (predict_label - actual_label) * derivative_softmax(outputLayer.input)
    delta_hidden = np.dot(outputLayer.weights.T, delta_output)*derivative_softmax(hiddenLayer.input)
    
    outputLayer.bias = outputLayer.bias - learning_rate*delta_output
    outputLayer.weights = outputLayer.weights - learning_rate*np.dot(delta_output, hiddenLayer.output.T)
    
    hiddenLayer.bias = hiddenLayer.bias - learning_rate*delta_hidden
    hiddenLayer.weights = hiddenLayer.weights - learning_rate*np.dot(delta_hidden, inputLayer.output.T)
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def trainNN(Network, datas):
    for i in range(datas.shape[0]):  # datas.shape[0]
        data = datas[i:i+1]
        label_index = data['label']
        pixel_data = data.drop(['label'], axis=1).values       
        actual_label = np.zeros((Network[-1].numberNodes, 1))
        actual_label[label_index.values[0], 0] = 1
        predict_label = predict(Network, pixel_data)
        Network = BP(Network, actual_label, predict_label)
    return Network

def testNN(Network, testdata):
    for i in range(5): # testdata.shape[0]
        # pixel_data = testdata[i:i+1].values
        # predict_label = predict(Network, pixel_data)
        # print(predict_label)
        data = testdata[i:i+1]
        label_index = data['label']
        pixel_data = data.drop(['label'], axis=1).values       
        actual_label = np.zeros((Network[-1].numberNodes, 1))
        actual_label[label_index.values[0], 0] = 1
        predict_label = predict(Network, pixel_data)
        print(np.argmax(predict_label))
 

def visiualization(datas):
    plt.figure(1)
    x = [151, 152, 153, 154, 155]
    for i in range(5):
        plt.subplot(x[i-5])
        # data = datas[i:i+1].values
        # pic = np.reshape(data, (28,28))
        data = datas[i:i+1]
        pixel_data = data.drop(['label'], axis=1).values
        pic = np.reshape(pixel_data, (28,28))
        plt.imshow(pic)


start = time.time()
# trainFile = "train.csv"
# data = pd.read_csv(trainFile)
# NN = createNN(784, 10, 3)
# np.save('NN0', NN)
# NN = trainNN(NN, data)
# np.save('NN1', NN)

testFile = "train.csv"
testdata = pd.read_csv(testFile)
visiualization(testdata)
NN0 = np.load('NN0.npy')
NN1 = np.load('NN1.npy')
testNN(NN1, testdata)

end = time.time()
print(end-start)
print("OK")
