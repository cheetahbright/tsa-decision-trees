

import importlib as imp

import clean_data
reload(clean_data)
from clean_data import *


cleaned_data = clean_data_f()


cleaned_data.shape
cleaned_data.columns[:5]
cleaned_data.set_index("Claim Number", inplace = True)


Y = cleaned_data["Close Amount"]
# Binary Y (1 if greater than 0, else 0 )
Y_bin = [1 if y > 0 else 0 for y in Y]
X = cleaned_data.drop(["Close Amount"], axis = 1)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y_bin, test_size = 0.3, random_state = 100)

from sklearn import tree
import pydot
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image


# This function creates images of tree models using pydot
def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names

    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return(graph)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)

path1 = clf.decision_path(X.loc[4].values.reshape(1,-1))
print(path1)


graph = print_tree(clf, X.columns)
Image(graph[0].create_png())
# Can see that if Disposition = Deny then the amount is  0

# How many depositions types are there ...
[x for x in X.columns if "Disposition" in x]
# 3 dispositions...

# Let's try using only the positive samples for a decision tree
# vs a linear regression

Y_pos = Y[Y >0]
X_pos = X.loc[Y_pos.index]

X_tr_pos, X_te_pos, y_tr_pos, y_te_pos =  train_test_split(X_pos, Y_pos, test_size = 0.3, random_state = 100)
pos_tree = tree.DecisionTreeRegressor(max_depth = 3, min_samples_leaf=30)
pos_tree.fit(X_tr_pos, y_tr_pos)
pos_tree.score(X_tr_pos, y_tr_pos)
pos_tree.score(X_te_pos, y_te_pos)

graph = print_tree(pos_tree, X.columns)
Image(graph[0].create_png())
feat_imps = pd.Series(pos_tree.feature_importances_, index = X.columns
feat_imps[feat_imps > 0]



## Try a linear model on the data

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize = True)
# normalize so we can compare coefficients
# Normalize as compared to standardized due to the categorical variables?
lin_reg.fit(X_tr_pos, y_tr_pos)
lin_reg.score(X_tr_pos, y_tr_pos)
lin_reg.score(X_te_pos, y_te_pos)
