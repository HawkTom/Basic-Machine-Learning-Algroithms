from sklearn.datasets import load_iris
from sklearn import tree
import os
iris = load_iris()
clf = tree.DecisionTreeClassifier()
lf = clf.fit(iris.data, iris.target)

with open(r"T:\iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

os.unlink('iris.dot')
