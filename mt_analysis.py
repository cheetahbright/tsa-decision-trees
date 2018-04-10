import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%matplotlib inline
import re

# https://www.dhs.gov/tsa-claims-data
os.getcwd()
path_check = input('Is this m or d?')
if path_check == 'm':
    os.chdir(r"C:\\Users\\board\\Desktop\\Kaggle\\tsa-decision-trees")
elif path_check =='d':
    os.chdir(r'C:\Users\drose\Documents\GitHub\tsa-decision-trees')
else:
    print("Ensure file is in path!")

data1 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")


### start data inspection ###
data1.shape
data1.head(100)
data1.describe()

col_n_unique = data1.nunique()
data2 = data1.replace("-", np.NaN)
col_n_missing = data2.isnull().sum()
col_stats = pd.concat([col_n_missing, col_n_unique], axis =1)
col_stats.columns = ["N_Missing", "N_Unique"]
col_stats.shape
col_stats

# Begin looking at Close Amount, dependent variable
y = data2["Close Amount"]
y.describe()
# Percent of 0's
len(y[y ==0])/len(y)

y75 = y[y != 0].quantile(0.75)
# Histogram of Close Amount
plt.xkcd()
y.hist(bins = 20)
y[y<y75].hist(bins = 20)

# Histogram of Close Amount without the 0 values
y[y!=0].hist(bins = 20)
plt.show()

### Create some features based on time
# Month, day of week for incident D and Date recieved
# Subtract incident d and date received
## Examine that subtraction per month and over time


data1.columns
data1["Date Received"], data1["Incident D"] = pd.DatetimeIndex(data1["Date Received"]), pd.DatetimeIndex(data1["Incident D"])

temp_date = pd.DatetimeIndex(data1["Date Received"])
temp_date2 = pd.DatetimeIndex(data1["Incident D"])



open_days = (temp_date - temp_date2).days
pd.Series(open_days).describe()
open_days_series = pd.Series(open_days)
plt.hist(open_days)
plt.show()


plt.hist(open_days_series[open_days_series < open_days_series.quantile(0.75)])
plt.title("Histogram of number of open days")
plt.ylabel("Number of occurances")
plt.xlabel("Number of days open")
plt.show()




#######################
#######################
### Begin One hot encoding process
#######################
#######################
# Start by getting an idea of how many variables we may be adding
data_clean = data2[pd.notnull(data2["Close Amount"])]
cols_for_dummies = ['Airport Code', 'Airport Name', 'Airline Name', 'Claim Type', 'Claim Site', 'Item Category', 'Disposition']
data2[cols_for_dummies].nunique().sum()
# One hot encoding
dummies = pd.get_dummies(data_clean[cols_for_dummies])
dummies.shape

###############

#######################################
#######################################
##### Final Data cleaning Prep ########
#######################################
#######################################
data_modeling = data_clean.drop(cols_for_dummies, axis = 1)
data_modeling.drop(["Date Received", "Incident D"], axis = 1, inplace = True)
data_modeling  = pd.concat([data_modeling, dummies], axis = 1)
#data_modeling.drop(["Date Received", "Incident D"], axis = 1, inplace = True) # Two sets of columns with this name
data_modeling.shape
data_modeling.head()
data_modeling.isnull().sum().sum()

data_modeling.dtypes.values

numb_bad_col_types = sum([True if col_type not in ["uint8", "float64", "int64"] else False for col_type in data_modeling.dtypes])
print("The number of columns that need checking are: ", numb_bad_col_types)
[col_type for col_type in data_modeling.dtypes if col_type not in ["uint8", "float64", "int64"]]
data_modeling.head()
'Close Amount' in list(data_modeling.columns)
[x for x in data_modeling.columns if "Date" in x]

X = data_modeling.drop(["Close Amount"], axis = 1)
#X = data_modeling.drop(["Close Amount", "Date Received", "Incident D"], axis = 1)
Y = data_modeling["Close Amount"]
X.columns

from sklearn import tree

try:
    import pydotplus as pydot
    print('yay')
except  ModuleNotFoundError:
    import pydot

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image

try:
    os.environ["PATH"] += os.pathsep + "C:/Users/board/Desktop/Kaggle/release/bin"
except:
    pass

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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

clf = tree.DecisionTreeRegressor(max_depth = 3, max_leaf_nodes=10, min_samples_leaf=10)
clf.fit(X_train,y_train)
clf.score(X_train, y_train)
clf.score(X_test,y_test)

from sklearn.model_selection import GridSearchCV
clf.get_params()
param_grid = [
  {'max_depth': [ 2, 5, 40, 100], 'max_leaf_nodes': [5, 20, 70, None], 'min_samples_leaf': [1, 5, 20, 100]},
  {'presort': [False, True], 'max_depth': [10,7]},
 ]

tree_reg = tree.DecisionTreeRegressor()
grid1 = GridSearchCV(tree_reg, param_grid)
grid1.fit(X_train, y_train)

grid1.best_params_
grid1.best_score_
grid1.cv_results_.keys()
results_dict = pd.DataFrame(grid1.cv_results_)
results_dict.head()
results_dict.sort_values(by="mean_test_score")


dir(grid1)


grid1.best_estimator_

means = grid1.cv_results_['mean_test_score']
means
stds = grid1.cv_results_['std_test_score']


graphs2, = print_tree(clf, features = X.columns)
Image(graphs2.create_png())


feat_importances = pd.DataFrame({"Features":X.columns, "Importances":clf.feature_importances_})
feat_importances.sort_values(by="Importances", ascending=False).head(10)
non_zero_importances = feat_importances[feat_importances["Importances"] > 0.0]
non_zero_feats = non_zero_importances["Features"].values
non_zero_feats

from sklearn.ensemble import RandomForestRegressor
clf2 = RandomForestRegressor(n_estimators=30, max_depth = 4)

clf2.fit(X_train, y_train)
clf2
clf2.score(X_train, y_train)
clf.score(X_train, y_train)

clf2.score(X_test, y_test)
clf.score(X_test, y_test)

y_pred = clf2.predict(X_test)
min(y_pred)
pd.Series(y_pred).describe()
y_test.describe()


y_pred_0_1 = [1 if x > 0 else 0 for x in y_pred]
y_actual_0_1 = [1 if x > 0 else 0 for x in y_test]

sum([1 if x== y else 0 for (x,y) in zip(y_pred_0_1, y_actual_0_1)])/len(y_pred_0_1)

zip(y_pred_0_1, y_actual_0_1)


from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=30, max_depth = 4)
y_train_2 = [1 if y > 0 else 0 for y in y_train]
y_test_2 = [1 if y > 0 else 0 for y in y_test]

clf3.fit(X_train, y_train_2)
clf2
clf3.score(X_train, y_train_2)
clf.score(X_train, y_train_2)

clf3.score(X_test, y_test_2)
clf.score(X_test, y_test)

y_pred = clf2.predict(X_test)
min(y_pred)
pd.Series(y_pred).describe()
