from sklearn.datasets import load_breast_cancer
from id3 import Id3Estimator
from id3 import export_graphviz
from graphviz import dot
import pydot
data = [['sunny',    'hot',   'high',   'weak'],
     ['sunny',    'hot',   'high',   'strong'],
     ['overcast', 'hot',   'high',   'weak'],
     ['rain',     'mild',  'high',   'weak'],
     ['rain',     'cool',  'normal', 'weak'],
     ['rain',     'cool',  'normal', 'strong'],
     ['overcast', 'cool',  'normal', 'strong'],
     ['sunny',    'mild',  'high',   'weak'],
     ['sunny',    'cool',  'normal', 'weak'],
     ['rain',     'mild',  'normal', 'weak'],
     ['sunny',    'mild',  'normal', 'strong'],
     ['overcast', 'mild',  'high',   'strong'],
     ['overcast', 'hot',   'normal', 'weak'],
     ['rain',     'mild',  'high',   'strong']]
target = ['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']
feature_names = ['outlook','temperature','humidity','wind']
estimator = Id3Estimator()
estimator.fit(data, target)
export_graphviz(estimator.tree_, 'tree.dot', feature_names)
print(estimator.predict([['rain', 'mild', 'high', 'strong']]))