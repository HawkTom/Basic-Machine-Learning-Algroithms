import RegressionTree as rt
import copy

def testError(tree, test_data, criterion = 'EOS'):
    pred, y = [], []
    error_square = 0
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
        error_square += pow(pred[-1] - y[-1], 2)
    if criterion == "EOS":
        return error_square
    elif criterion == "CC":
        return rt.np.corrcoef(pred, y, rowvar=0)
    else:
        return "The method is not exist."

def Pruning_node(tree):
    node_left = tree.leftChild
    node_right = tree.rightChild
    if node_right == None:
        return "OK"
    if node_left.leftChild == None and node_right.leftChild == None:
        tree.amount = (node_right.amount + node_left.amount)
        tree.estimate = (node_left.amount * node_left.estimate + node_right.amount * node_right.estimate) / tree.amount
        tree.leftDepth, tree.rightDepth = 0, 0
        tree.leftChild, tree.rightChild = None, None
    else:
        if tree.leftDepth >= tree.rightDepth:
            statues = Pruning_node(node_left)
        else:
            statues = Pruning_node(node_right)
        tree.leftDepth = max(tree.leftChild.leftDepth, tree.leftChild.rightDepth) + 1
        tree.rightDepth = max(tree.rightChild.rightDepth, tree.rightChild.leftDepth) + 1
        if statues == "OK":
            return "OK"

    return tree

def Pruning(tree, testData):
    min_error = float('Inf')
    tree_temp = tree
    while tree_temp.amount == 0:
        tree_temp1 = Pruning_node(tree_temp)

        # rt.dot_File(tree_temp1, "pruning_temp.dot")
        # file_path = rt.os.getcwd()
        # rt.os.system(file_path[0:2] + "\n")
        # rt.os.system("cd " + file_path[3:] + "\\")
        # rt.os.system("dot -Tpdf pruning_temp.dot -o pruning_temp.pdf")
        # rt.os.system("start pruning_temp.pdf")

        error = testError(tree_temp, test_data, criterion='EOS')
        if  error < min_error:
            min_error = error
            ans = copy.deepcopy(tree_temp)
        tree_temp =tree_temp1

    return ans

train_dataFile = "train.txt"
test_dataFile = "test.txt"

# output the train data and test data from the file
data = rt.dataRead(train_dataFile)
test_data = rt.dataRead(test_dataFile)



# create the regression tree by the data
x = rt.createTree(data)
rt.dot_File(x, "regression.dot")
rt.plot_model(1, data)   # plot the line and regression tree model in th graph
# pruning
t = Pruning(x, test_data)
rt.plot_model(2, test_data, []) # plot the line and regression tree model in th graph
rt.dot_File(t, "tree_end.dot")

# output the tree information to dot file and pdf file
file_path = rt.os.getcwd()
rt.os.system(file_path[0:2] + "\n")
rt.os.system("cd " + file_path[3:] + "\\")
rt.os.system("dot -Tpdf tree_end.dot -o tree_end.pdf")
rt.os.system("dot -Tpdf regression.dot -o regression.pdf")
rt.os.system("start tree_end.pdf")
rt.os.system("start regression.pdf")


rt.plt.show()