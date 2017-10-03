from math import log
import os

class Tree(object):

    def __init__(self, f):
        self.attr = f
        self.next = []

    def dataGet(self, data):
        self.set = data
        self.size = len(data)

    def fClassify(self, feature):
        self.classify = feature


# data pre-processing
def DataPre(FilePath, features):
    data_feature = []
    with open(FilePath, 'r') as f:
        for line in f:
            rowDict = {}
            words = line.strip().split('\t')
            for i in range(len(features)):
                rowDict[features[i]] = words[i]
            data_feature.append(rowDict)
    return data_feature


# calculating the information entropy
def EntropyCal(labels):
    n = len(labels)
    count = {}
    entropy = 0
    for label in labels:
        if label not in count:
            count[label] = 1
        else:
            count[label] += 1
    for label in count:
        p = count[label]/n
        entropy -= p*log(p, 2)
    return entropy


# Information Gain Calculating
def InfoGain(data_features, features, classify_type):
    IGain, Gain_ratio ={}, {}
    Entropy = EntropyCal([feature['label'] for feature in data_features])
    # print(Entropy)
    for feature in features[:-1]:
        count = {}
        IGain[feature] = 0
        for sample in data_features:
            if sample[feature] not in count:
                count[sample[feature]] = [sample['label']]
            else:
                count[sample[feature]].append(sample['label'])
        # Conditional Entropy
        CEntropy, IV = 0, 0
        for attribute in count:
            temp = len(count[attribute]) / len(data_feature)
            CEntropy += EntropyCal(count[attribute])* temp
            if classify_type == "C4.5":
                IV -= temp*log(temp, 2)
        IGain[feature] =  Entropy - CEntropy
        if classify_type == "C4.5":
            Gain_ratio[feature] = IGain[feature] / IV
    # print(IGain)
    if classify_type == "ID3":
        return max(IGain.items(), key=lambda x:x[1])[0]
    elif classify_type =="C4.5":
        return max(Gain_ratio.items(), key=lambda x: x[1])[0]
    else:
        print("The Only Two Way is \"ID3\" and \"C4.5\"")
        return False

# partition samples
def Partition(sample, attribute):
    ans = {}
    same_attribute = []
    for individual in sample:
        individual_attribute = individual[attribute]
        individual.pop(attribute)
        if individual_attribute not in same_attribute:
            same_attribute.append(individual_attribute)
            ans[individual_attribute] = [individual]
        else:
            ans[individual_attribute].append(individual)
    return ans


# whether the classify finish or not
def labelSame(dataSet):
    count = {}
    for data in dataSet:
        if data['label'] not in count:
            count[data['label']] = 1
        else:
            count[data['label']] += 1
    if len(count) > 1:
        return False
    else:
        return True



def creatTree(data, attr, features, classify_type = "ID3"):
    if labelSame(data) or len(features) == 1:
        t = Tree(attr)
        t.dataGet(data)
        t.fClassify(None)
        return t
    classify_feature = InfoGain(data, features, classify_type)
    t = Tree(attr)
    t.dataGet(data)
    t.fClassify(classify_feature)
    features_temp = [i for i in features]
    features_temp.remove(classify_feature)
    Children = Partition(data, classify_feature)
    for Child in Children:
        t.next.append(creatTree(Children[Child], Child, features_temp))
    return t

command = []
def plot_tree(tree):
    if len(tree.next) == 0:
        return "OK"
    global command
    for node in tree.next:
        command.append(tree.attr+str(tree.size)+ "->"+node.attr+str(node.size))
        plot_tree(node)


def dot_File(tree, output_file):
    plot_tree(tree)
    with open(output_file, 'w') as f:
        data = "digraph G{" + "\n\t" + "\n\t".join(command) + "\n}"
        f.write(data)


data_file_path = r"lenses.txt"
output_file_dot = "lense.dot"
output_file_pdf = "lense.pdf"
features = ['age', 'prescript', 'astigmatic', 'tearRate', 'label']
data_feature = DataPre(data_file_path, features)
D = creatTree(data_feature, 'all', features,classify_type="C4.5")
dot_File(D, output_file_dot)

try:
    file_path = os.getcwd()
    os.system(file_path[0:2] + "\n")
    os.system("cd " + file_path[3:] + "\\")
    os.system("dot -Tpdf " + output_file_dot + " -o " + output_file_pdf)
    os.system("start " + output_file_pdf)
except :
    print(" The graphviz is not installed. ")



# ??dot?? ?graphviz ?????
# ??? dot -Tpdf tree.dot -o output.pdf
