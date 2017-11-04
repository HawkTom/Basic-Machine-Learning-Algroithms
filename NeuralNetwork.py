import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


class layer(object):
    def __init__(self, numberNodes_ThisLayer):
        self.numberNodes = numberNodes_ThisLayer
        self.input = np.zeros((numberNodes_ThisLayer, 1))

    def activate(self):
        self.output = Sigmoid(self.input)

    def createParameters(self, numberNods_LastLayer):
        self.weights = np.random.randn(self.numberNodes, numberNods_LastLayer)
        self.bias = np.random.randn(self.numberNodes, 1)


def Sigmoid(x):
    sm = 1 / (np.exp(-x) + 1)
    return sm

def derivative_sigmoid(x):
    return Sigmoid(x)*(1- Sigmoid(x))

def ReLu(x, derivative=False):
    if (derivative==False):
        return x*(x>0)
    else:
        return 1*(x>0)

def Softmax(x):
    sfm = np.exp(x)/sum(np.exp(x))
    return sfm


def createNN(inputSize, outputSize, numberLayers):
    hiddenSize = 35
    inputLayer = layer(inputSize)
    hiddenLayer = layer(hiddenSize)
    hiddenLayer.createParameters(inputSize) # weights: 35x784   bias: 35x1
    outputLayer = layer(outputSize)
    outputLayer.createParameters(hiddenSize)# weights: 10x35    bias:10x1
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def forward(Network, data):
    inputLayer = Network[0]
    inputLayer.output = data.T
    # inputLayer.activate()
    
    hiddenLayer = Network[1]
    hiddenLayer.input = np.dot(hiddenLayer.weights, inputLayer.output) + hiddenLayer.bias
    hiddenLayer.activate()
    
    outputLayer = Network[2]
    outputLayer.input = np.dot(outputLayer.weights, hiddenLayer.output) + outputLayer.bias
    outputLayer.activate()

    return outputLayer.output


def BP(Network, actual_label, predict_label, learning_rate = 0.2):
    inputLayer, hiddenLayer, outputLayer = Network[0], Network[1], Network[2]
    
    alpha = learning_rate

    delta_output = (predict_label - actual_label) * derivative_sigmoid(outputLayer.input)
    delta_hidden = np.dot(outputLayer.weights.T, delta_output)*derivative_sigmoid(hiddenLayer.input)
    
    outputLayer.bias = outputLayer.bias - alpha * delta_output
    outputLayer.weights = outputLayer.weights - alpha * np.dot(delta_output, hiddenLayer.output.T)
    
    hiddenLayer.bias = hiddenLayer.bias - alpha * delta_hidden
    hiddenLayer.weights = hiddenLayer.weights - alpha * np.dot(delta_hidden, inputLayer.output.T)
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network


def trainNN(Network, datas):
    for j in range(2):
        for i in range(datas.shape[0]):  # datas.shape[0]
            data = datas[i:i+1]
            label_index = data['label']
            pixel_data = data.drop(['label'], axis=1).values/255     
            actual_label = np.zeros((Network[-1].numberNodes, 1))
            actual_label[label_index.values[0], 0] = 1
            predict_label = forward(Network, pixel_data)
            error = np.sqrt(sum((predict_label-actual_label)**2))
            print(error)
            Network = BP(Network, actual_label, predict_label)
    return Network

def trainNN_BatchGD(Network, datas):
    # datas = datas[0:5]
    label_index = datas['label'].values 
    pixel_data = datas.drop(['label'], axis=1).values   #pixel data: 42000*784
    actual_label = np.zeros((Network[-1].numberNodes, datas.shape[0]))
    for i in range(datas.shape[0]):
        actual_label[label_index[i], i] = 1   # actual label: 10x42000
    predict_label = forward(Network, pixel_data) # predict label: 10x42000
    print(predict_label.shape)
    # Network = BP(Network, actual_label, predict_label)
    return 0

def testNN(Network, testdata):
    result = []
    for i in range(testdata.shape[0]): # testdata.shape[0]
        pixel_data = testdata[i:i+1].values
        predict_label = forward(Network, pixel_data)
        # data = testdata[i:i+1]
        # label_index = data['label']
        # pixel_data = data.drop(['label'], axis=1).values       
        # actual_label = np.zeros((Network[-1].numberNodes, 1))
        # actual_label[label_index.values[0], 0] = 1
        # predict_label = forward(Network, pixel_data)
        predict_num = np.argmax(predict_label)
        result.append(str(i+1) + ',' + str(predict_num))
    return result
        #print(np.argmax(predict_label))
 


def visiualization(datas):
    x = [151, 152, 153, 154, 155]
    for i in range(25,30):
        plt.subplot(x[i-25])
        data = datas[i:i+1].values
        pic = np.reshape(data, (28,28))
        # data = datas[i:i+1]
        # pixel_data = data.drop(['label'], axis=1).values
        # pic = np.reshape(pixel_data, (28,28))
        plt.imshow(pic)


start = time.time()
#trainFile = "train.csv"
#data = pd.read_csv(trainFile)
#NN = createNN(784, 10, 3)
## trainNN_BatchGD(NN, data)
#
#
#NN = createNN(784, 10, 3)
#NN = trainNN(NN, data)
#np.save('NN1', NN)

testFile = "test.csv"
testdata = pd.read_csv(testFile)
#visiualization(testdata)
NN1 = np.load('NN1.npy')
result_ = testNN(NN1, testdata)
with open("result.csv",'w') as f:
    f.write('ImageId,Label\n'+'\n'.join(result_))

end = time.time()
print(end-start)
print("OK")
