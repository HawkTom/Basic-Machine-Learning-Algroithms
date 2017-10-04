# import matplotlib.pyplot as plt
# import numpy as np

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
            if i != point:
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
        if point != split_point:
            if point[0] <= split_point[0]:
                count['-'].append(point)
            else:
                count['+'].append(point)
    print(len(count['+']))
    print(len(count['-']))
    return count




dataFile = "train.txt"
data = dataRead(dataFile)
print(Partition(data))


# a = np.array(set)
# plt.figure(0)
# plt.plot(a[:,0],a[:,1],'.',color = 'green')
# plt.show()
# print(set)
