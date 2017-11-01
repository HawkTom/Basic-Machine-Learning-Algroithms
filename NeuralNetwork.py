import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class layer(object):
    def __init__(self, numberNodes_ThisLayer):
        self.numberNodes = numberNodes_ThisLayer
        self.input = np.zeros((numberNodes_ThisLayer, 1))

    def activate(self):
        self.output = Sigmoid(self.input)

    def createWeights(self, numberNods_LastLayer):
        self.weights = np.random.random(
            (numberNods_LastLayer, self.numberNodes))


def Sigmoid(x):
    sm = 1 / (np.exp(-x) + 1)
    return sm


def createNN(inputSize, outputSize, numberLayers):
    hiddenSize = 100
    inputLayer = layer(inputSize)
    hiddenLayer = layer(hiddenSize)
    hiddenLayer.createWeights(inputSize)
    outputLayer = layer(outputSize)
    outputLayer.createWeights(hiddenSize)
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def predict(Network, data):
    pass


def BP(Network, actual_label, predict_label):
    pass
    return Network


def trainNN(Network, datas):
    for i in range(datas.shape[0]+1):
        data = datas[i:i+1]
        label_index = data['label']
        pixel_data = data.drop(['label'], axis=1)
        actual_label = np.zeros(1, Network[0].numberNodes)
        actual_label[label_index[str(i)]] = 1
        predict_label = predict(Network, pixel_data)
        Network = BP(Network, actual_label, predict_label)
    return Network


trainFile = "train.csv"
data = pd.read_csv(trainFile)

print(data)
