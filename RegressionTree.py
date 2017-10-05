import matplotlib.pyplot as plt
import numpy as np
import os
ErrorPermit = 1

class Tree(object):
    def __init__(self, separate_data=0, estimate_data=0):
        self.val = separate_data
        self.estimate = estimate_data
        self.leftChild = None
        self.rightChild = None

def dataRead(File):
    with open(File) as f:
        dataSet = []
        for line in f:
            s = line.split('\t')
            data = [float(s[0]),float(s[1])]
            dataSet.append(data)
    dataSet = sorted(dataSet)
    # print(dataSet)
    return dataSet


def errorCal(count):
    error_square = 0
    for key in count:
        if len(count[key]) != 0:
            avr = sum(count[key]) / len(count[key])
            error_square += sum([pow(i-avr, 2) for i in count[key]])
    return error_square


def split_data(dataSet):
    error, j = {}, 0
    for point in dataSet:
        j += 1
        count = {'+': [], '-': []}
        for i in dataSet:
            if i[0] <= point[0]:
                count['-'].append(i[1])
            else:
                count['+'].append(i[1])
        error[j] = errorCal(count)
    split_point = int(min(error.items(), key=lambda x: x[1])[0])
    return split_point


def Partition(dataSet):
    split_point = dataSet[split_data(dataSet)]
    count = {'+':[],'-':[]}
    for point in dataSet:
        if point[0] <= split_point[0]:
            count['-'].append(point)
        else:
            count['+'].append(point)
    # print(len(count['+']))
    # print(len(count['-']))
    return count, split_point

def stopCondition(dataList):
    dataList = [i[1] for i in dataList]
    avr = sum(dataList) / len(dataList)
    error_square = sum([pow(i - avr, 2) for i in dataList])
    return error_square, avr

def createTree(dataSet):
    error = stopCondition(dataSet)
    ans = Partition(dataSet)
    dataDict = ans[0]
    split_point = ans[1]
    if error[0] <= ErrorPermit or len(dataDict['+'])==0 or len(dataDict['-']) == 0:
        return Tree(estimate_data = error[1])

    head = Tree(separate_data=split_point[0])
    head.leftChild = createTree(dataDict['+'])
    head.rightChild = createTree(dataDict['-'])

    return  head

command =[]
model_point = []
def plot_tree(tree):
    global command, model_point
    tree_left = tree.leftChild
    tree_right = tree.rightChild

    if tree_left.leftChild == None:
        model_point.append([tree.val, tree_left.estimate])
        command.append(str(tree.val) + '->' + str(tree_left.estimate) + '[label= \">' + str(tree.val) + '\"]')
    else:
        command.append(str(tree.val) + '->' + str(tree_left.val) + '[label= \">' + str(tree.val) + '\"]')
        plot_tree(tree.leftChild)

    if tree_right.leftChild == None:
        model_point.append([tree.val-1, tree_right.estimate])
        command.append(str(tree.val) + '->' + str(tree_right.estimate) + '[label= \"<' + str(tree.val) + '\"]')
    else:
        command.append(str(tree.val) + '->' + str(tree_right.val) + '[label= \"<' + str(tree.val) + '\"]')
        plot_tree(tree.rightChild)

    return "OK"

def dot_File(tree, output_file):
    plot_tree(tree)
    with open(output_file, 'w') as f:
        data = "digraph G{" + "\n\t" + "\n\t".join(command) + "\n}"
        f.write(data)

def plot_model(tree):
    model_temp = []
    for point in model_point:
        model_temp.append([point[0] + 1, point[1]])

    model = []
    for i in range(len(model_point)):
        model.append(model_temp[i])
        model.append(model_point[i])

    a = np.array(model[::-1])
    b = np.array(data)
    fig = plt.figure(0)
    axes = fig.add_subplot(111)
    axes.plot(a[:, 0], a[:, 1], '-', color='green')
    axes.plot(b[:, 0], b[:, 1], '.', color='red')
    axes.set_xticks(list(range(23)))
    plt.show()

if __name__ == "__main__":
    dataFile = "train.txt"
    output_file_dot = "regression.dot"
    output_file_pdf = "regression.pdf"

    data = dataRead(dataFile)  # output the train data from the file
    x = createTree(data) # create the regression tree by the data
    dot_File(x, output_file_dot) # output the tree information to dot file
    plot_model(x) # plot the line and regression tree model in th graph

    # # output the tree model in the pdf
    # file_path = os.getcwd()
    # os.system(file_path[0:2] + "\n")
    # os.system("cd " + file_path[3:] + "\\")
    # os.system("dot -Tpdf " + output_file_dot + " -o " + output_file_pdf)
    # os.system("start " + output_file_pdf)







