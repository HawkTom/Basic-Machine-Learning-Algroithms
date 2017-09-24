from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree


data_file_path = r"T:\MyStudyData\class\Meachine Learning\Experiment\lab1 decision tree\lenses.txt"
with open(data_file_path, 'r') as f:
    data_feature = []
    features = ['age','prescript','astigmatic','tearRate']
    data_labels = []
    for line in f:
        rowDict = {}
        words = line.strip().split('\t')
        data_labels.append(words.pop())
        for i in range(4):
            rowDict[features[i]] = words[i]
        data_feature.append(rowDict)


vec = DictVectorizer()
dummyX = vec.fit_transform(data_feature).toarray()

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(data_labels)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(dummyX, dummyY)


with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(),out_file=f)


# 输出dot文件 用graphviz 工具可视化
# 命令行 dot -Tpdf tree.dot -o output.pdf