import RegressionTree as rt

def testError(test_data, tree):
    pred = []
    y = []
    for data in test_data:
        y.append(data[1])
        head = tree
        while True:
            if head.leftChild == None:
                pred.append(head.estimate)
                break
            if data[0] > head.val:
                head = head.leftChild
            else:
                head = head.rightChild
    return rt.np.corrcoef(pred, y, rowvar=0)

def Pruning(tree, testData):
    pass

train_dataFile = "train.txt"
output_file_dot = "regression.dot"
output_file_pdf = "regression.pdf"

data = rt.dataRead(train_dataFile)  # output the train data from the file
x = rt.createTree(data) # create the regression tree by the data
rt.dot_File(x, output_file_dot) # output the tree information to dot file
rt.plot_model(1, data) # plot the line and regression tree model in th graph

test_dataFile = "test.txt"
test_data = rt.dataRead(test_dataFile)
rt.plot_model(2, test_data, [])

print(testError(test_data, x))


# rt.plt.show()