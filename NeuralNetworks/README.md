
## Supervise Learning Report

### Pre-Lab

**Problem Definition**

- Given the CIFAR-10 Dataset, using a supercised learning method to get a classification model.  
- Train data includes 50000 images, and test data includes 10000 iamges.


**Problem Analysis**

This classification task is more difficult than hand-writing digits recognition. Because each image is colourful and the similar in color may has different label. And we can not only classify them by the color feature. We must need collect more information about the picture.

The picture below is the result analysised by a technic called t-SNE. It shows the difference among samples. Much closer, much more similar.  It is obviously that two images that has similar color features has much probability of in different labels.

<img src="http://cs231n.github.io/assets/pixels_embed_cifar10.jpg">

**Potential Method**

- Neural Network. Neural Network is a very useful way to train a classifiaction model, but it is complex and needs much time to train the parameters in the model. 
- K-Nearest-Neighbour. KNN is a very simple way to get a model, the label of a test sample depends on its distance to all the train samples. Voting among K nearest train samples dermeters the predict label of test sample. The time on traing is very short, but the time of predicting will be very long.  

**Method Chosen**

Finally, I choose the neural network to train the classification model. 

### Code Explanation

**File structure**

```
lab8
├── data
│   └── cifar-10-batches-py    
│       ├── banches.meta
│       ├── data_batch_1
│       └── ...
├── cifar10.py               
├── lab8.py                   
└── main.py 
```

cifar10.py and main.py has been given, so the algorithm is in the lab8.py.

**Code Comment and result**


```python
import numpy as np
```

- Simoid activate function: 
  $y = \frac{1}{1+e^{-a}}$
- Sigmoid derivative functin:
  $y = y(1-y)$
- Softmx activate fucntion:
  $y = \frac{e^{x_i}}{\sum e^{x_i}}$


```python
def Sigmoid(x):
    sm = 1 / (np.exp(-x) + 1)
    return sm

def derivative_sigmoid(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

def Softmax(x):
    sfm = np.exp(x) / sum(np.exp(x))
    return sfm
```

**Class of network layer** 

The class include layers node, active function, and parameters


```python
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
```

**Create Neural Network**

The network only contains 3 layers. It is very simple.


```python
def createNN(inputSize, outputSize, numberLayers):
    hiddenSize = 33
    inputLayer = layer(inputSize)
    hiddenLayer = layer(hiddenSize)
    hiddenLayer.createParameters(inputSize)  
    outputLayer = layer(outputSize)
    outputLayer.createParameters(hiddenSize)  
    Network = [inputLayer, hiddenLayer, outputLayer]
    return Network
```

**Forward process**

input a vector whose size is 3072x1 

output a result of classification with size of 10x1


```python
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
```

**Back Propagation**

This is the key algorithm to train a network well. The basis of algorithm is chain rule and gradient descent. 

BP algorithm is a little complex. The details will not show at this report. But there is a document which explain the procedure well. You can click [here](https://hawktom.github.io/Homework/notes/BackPropagation.pdf). 

The input is actual_label, predict_label and learning rate. 
 - the actual_label and predict_label both are vector
 - the leanirng rate is default 0.5, and also it should be regulated to get a better performance.


```python
def BP(Network, actual_label, predict_label, learning_rate=0.1):
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
```

**Training function**

This function mainly to preprocess the raw data and call the function corresponding to the each step including creating, forwarding and back propagation

Train the neural neural network by mini-batch.


```python
def train_Batch(images, one_hot_labels):
    NN = createNN(3072, 10, 3)
    batch_number = 2500
    batch_size = int(len(images) / batch_number)
    for j in range(5):
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
```

**Predicting**

Predicting actually is a forward process. When we get a test image, we only need to forward the data in the trained model to get its label. 

Predicting is very efficient. This is the advantage of neural network. 


```python
def predict(images, NN):
    print(images.shape)
    labels = np.zeros((len(images),), dtype=int)
    for i in range(len(images)):
        img = images[i]
        img = np.reshape(img, (1, 32 * 32 * 3))
        predict_label = forward(NN, img)
        labels[i] = np.argmax(predict_label)
    # Return a Numpy ndarray with the length of len(images).
    # e.g. np.zeros((len(images),), dtype=int) means all predictions are 'airplane's
    print(labels)
    return labels
```

### Result

**Parameter:**
- leaning rate: 0.8
- hidden layer nodes: 60
- epoch: 20
- batch size: 500
- loss function: entropy cross

**Result:** accuracy: 27.85%


