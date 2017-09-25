from math import log



# data pre-processing
def DataPre(FilePath, features):
    data_feature = []
    with open(FilePath, 'r') as f:
        for line in f:
            rowDict = {}
            words = line.strip().split('\t')
            for i in range(5):
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
def InfoGain(data_features, features):
    IGain ={}
    Entropy = EntropyCal([feature['label'] for feature in data_features])
    print(Entropy)
    for feature in features[:-1]:
        count = {}
        IGain[feature] = 0
        for sample in data_features:
            if sample[feature] not in count:
                count[sample[feature]] = [sample['label']]
            else:
                count[sample[feature]].append(sample['label'])
        # Conditional Entropy
        CEntropy = 0
        for attribute in count:
            CEntropy += EntropyCal(count[attribute])*len(count[attribute])/len(data_features)
        IGain[feature] =  Entropy - CEntropy
    print(IGain)
    return max(IGain.items(), key=lambda x:x[1])[0]

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


DTree = []
def creatTree(data, features):
    global DTree
    if labelSame(data) or len(features) == 1:
        DTree.append(data)
        return "OK"
    classify_feature = InfoGain(data, features)
    features.remove(classify_feature)
    Children = Partition(data, classify_feature)
    for Child in Children:
        creatTree(Children[Child],features)



data_file_path = r"T:\MyStudyData\class\Meachine Learning\Experiment\lab1 decision tree\lenses.txt"
features = ['age', 'prescript', 'astigmatic', 'tearRate', 'label']
data_feature = DataPre(data_file_path, features)
creatTree(data_feature, features)

# print(Partition(data_feature,'tearRate'))
