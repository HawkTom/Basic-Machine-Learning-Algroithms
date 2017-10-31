import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class layer(object):
    
    def __init__(self, numberNodes_ThisLayer):
        self.numberNodes = numberNodes_ThisLayer
        self.input = np.zeros((numberNodes_ThisLayer, 1))
        self.activate()
    
    def activate(self):
        self.output = Sigmoid(self.input)
    
    def createWeights(self, numberNods_LastLayer):
        self.weights = np.random.random((numberNods_LastLayer, self.numberNodes))
        

def Sigmoid(x):
    sm = 1/(np.exp(-x)+1)
    return sm



x = np.arange(20)-10
sigmoid = Sigmoid(x)
print(sigmoid)

plt.plot(x, sigmoid)
plt.show()

# trainFile = "train.csv"
# data = pd.read_csv(trainFile)
#
# print(data)