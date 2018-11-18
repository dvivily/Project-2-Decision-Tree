import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# from sklearn import tree
# import pydotplus
# import sys
# import os       
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

path = 'data.csv'
df = pd.read_csv(path)
df['type']=df['type'].map(str.strip)
mapping_dictionary = {"type":{'van': 1, 'saab': 1, 'bus': -1, 'opel': -1}}
df = df.replace(mapping_dictionary)
print(df.head(10))
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
def get_arrays(df):
    x = np.array(df.iloc[:,:-1])
    y = np.array(df.iloc[:,-1])
    return x, y
df_train_x, df_train_y = get_arrays(df_train)
df_test_x, df_test_y = get_arrays(df_test)
# print(df_train_x)
# print(df_train_y)

clf = DecisionTreeClassifier(max_depth=7)
clf = clf.fit(df_train_x, df_train_y)
train_y_pred = clf.predict(df_train_x)
test_y_pred = clf.predict(df_test_x)
tree_train = accuracy_score(df_train_y, train_y_pred)
tree_test = accuracy_score(df_test_y, test_y_pred)
tree_recall_score = recall_score(df_test_y, test_y_pred, average='macro')
tree_metrics = metrics.precision_score(df_test_y, test_y_pred, average='macro')
tree_f1 = metrics.f1_score(df_test_y, test_y_pred, average='weighted')
print('df_test_y:',df_test_y)
print('test_y_pred:',test_y_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test)) 
print('Decision tree train/test recall %.3f' % tree_recall_score)
print('Decision tree train/test precision %.3f' % tree_metrics)
print('Decision tree train/test f1_score %.3f' % tree_f1)

#决策树可视化
# dot_data = tree.export_graphviz(clf, out_file=None) 
# graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_pdf("tree.pdf")


clf_RF = RandomForestClassifier(n_estimators=180, criterion='gini', max_features='auto', max_depth=None, min_samples_split=2, bootstrap=True, n_jobs=1, random_state=1)
clf_RF = clf_RF.fit(df_train_x, df_train_y)
train_y_pred = clf_RF.predict(df_train_x)
test_y_pred = clf_RF.predict(df_test_x)
clf_RF_train = accuracy_score(df_train_y, train_y_pred)
clf_RF_test = accuracy_score(df_test_y, test_y_pred)
RF_recall_score = recall_score(df_test_y, test_y_pred, average='macro')
RF_metrics = metrics.precision_score(df_test_y, test_y_pred, average='macro')
RF_f1 = metrics.f1_score(df_test_y, test_y_pred, average='weighted') 
print('Random Forest train/test accuracies %.3f/%.3f' % (clf_RF_train, clf_RF_test)) 
print('RandomForest recall %.3f' % RF_recall_score)
print('RandomForest precision %.3f' % RF_metrics)
print('RandomForest f1_score %.3f' % RF_f1)

clf_GaussianNB = GaussianNB()
clf_GaussianNB = clf_GaussianNB.fit(df_train_x, df_train_y)
train_y_pred = clf_GaussianNB.predict(df_train_x)
test_y_pred = clf_GaussianNB.predict(df_test_x)
clf_GaussianNB_train = accuracy_score(df_train_y, train_y_pred)
clf_GaussianNB_test = accuracy_score(df_test_y, test_y_pred)
GaussianNB_recall_score = recall_score(df_test_y, test_y_pred, average='macro')
GaussianNB_metrics = metrics.precision_score(df_test_y, test_y_pred, average='macro')
GaussianNB_f1 = metrics.f1_score(df_test_y, test_y_pred, average='weighted') 
print('GaussianNB train/test accuracies %.3f/%.3f' % (clf_GaussianNB_train, clf_GaussianNB_test))
print('GaussianNB recall %.3f' % GaussianNB_recall_score)
print('GaussianNB precision %.3f' % GaussianNB_metrics)
print('GaussianNB f1_score %.3f' % GaussianNB_f1) 

# clf_SVC = SVC(gamma='auto')
# clf_SVC = clf_SVC.fit(df_train_x, df_train_y)
# train_y_pred = clf_SVC.predict(df_train_x)
# test_y_pred = clf_SVC.predict(df_test_x)
# clf_SVC_train = accuracy_score(df_train_y, train_y_pred)
# clf_SVC_test = accuracy_score(df_test_y, test_y_pred)
# SVC_recall_score = recall_score(df_test_y, test_y_pred, average='macro')
# SVC_metrics = metrics.precision_score(df_test_y, test_y_pred, average='macro')
# SVC_f1 = metrics.f1_score(df_test_y, test_y_pred, average='weighted') 
# print('SVM train/test accuracies %.3f/%.3f' % (clf_SVC_train, clf_SVC_test)) 
# print('SVC recall %.3f' % SVC_recall_score)
# print('SVC precision %.3f' % SVC_metrics)
# print('SVC f1_score %.3f' % SVC_f1)




