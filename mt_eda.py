import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%matplotlib inline
import re


os.getcwd()
os.chdir(r"C:\\Users\\board\\Desktop\\Kaggle\\tsa-decision-trees")
data1 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")

data1.shape
data1.head(100)
data1.describe()

col_n_unique = data1.nunique()

data2 = data1.replace("-", np.NaN)
col_n_missing = data2.isnull().sum()
col_n_missing.shape
col_n_missing

col_stats = pd.concat([col_n_missing, col_n_unique], axis =1)
col_stats.columns = ["N_Missing", "N_Unique"]
col_stats.shape
col_stats

# Begin looking at CLose Amount
y = data2["Close Amount"]
y.describe()
# Percent of 0's
len(y[y ==0])/len(y)

# Histogram of Close Amount
y.hist(bins = 20)

# Histogram of Close Amount without the 0 values
y[y!=0].hist(bins = 20)

### Quick time Series analysis

# TODO: Dates analysis
data2.head()
data2.columns
data2["Date Received"], data2["Incident D"] = pd.DatetimeIndex(data2["Date Received"]), pd.DatetimeIndex(data2["Incident D"])

data2["Date Received"].describe()
data2["Incident D"].describe()

sorted_DR = data2.sort_values(by="Date Received")
sorted_DR = sorted_DR.set_index("Date Received")
plt.plot(sorted_DR["Date Received"], sorted_DR["Close Amount"])

monthly_group = sorted_DR.groupby(pd.Grouper(freq = 'M'))
monthly_group["Close Amount"].describe()


# Close amount average by month
plt.plot(monthly_group["Close Amount"].describe()["mean"])

numb_0_by_month = monthly_group["Close Amount"].apply(lambda x: len(x[x==0]))
numb_cnt_by_month = monthly_group["Close Amount"].count()

plt.plot(numb_0_by_month.index, numb_0_by_month, 'r--',  numb_cnt_by_month.index, numb_cnt_by_month, 'g--')


"""
# Combine these plx
# Number of 0's per month
numb_0_line, = plt.plot(numb_0_by_month.index, numb_0_by_month, 'r--', label = "Number of 0's")
numb_cnt_line, = plt.plot(numb_cnt_by_month.index, numb_cnt_by_month, 'g--', label = "Total Count")
plt.legend(handles =[numb_0_line, numb_cnt_line])
ax.set_title("Number of 0s and Total Count vs Month")
plt.show()
"""

### Create some features based on time
# Month, day of week for incident D and Date recieved
# Subtract incident d and date received
## Examine that subtraction per month and over time



### Begin One hot encoding process
# Start by getting an idea of how many variables we may be adding
data_clean = data2[pd.notnull(data2["Close Amount"])]
cols_for_dummies = ['Airport Code', 'Airport Name', 'Airline Name', 'Claim Type', 'Claim Site', 'Item Category', 'Disposition']



data2[cols_for_dummies].nunique().sum()


# One hot encoding
dummies = pd.get_dummies(data_clean, columns = cols_for_dummies)
dummies.shape
dummies.head()
data_clean.head()


dummy_cols = dummies.columns
dummy_cols[-4:]
dummy_cols.shape

dict1 = {}
for col_base in cols_for_dummies:
    dict1[col_base] = [1 if col_base in d_col_name else 0 for d_col_name in dummy_cols]

# Compare the number of unique in pd dummies data set vs. # uniques in original data set
{k:sum(v) for k,v in dict1.items()}

data2[cols_for_dummies].nunique()

# I suspect some of the names have spaces which did not work well for the naming convention
# Let's confirm


uniques_dataset = {col:data2[col].unique() for col in cols_for_dummies}
uniques_dataset.keys()
uniques_dataset["Claim Type"]

## pd get dummy unique col names
pd_uniques = {}
for base_col in cols_for_dummies:
    pd_uniques[base_col] = [re.sub(base_col+"_", "", col) for col in dummy_cols if base_col in col]
# Validate
{k:len(v) for k,v in pd_uniques.items()}

# CHeck that it matches the list before
{k:len(v) for k,v in pd_uniques.items()} == {k:sum(v) for k,v in dict1.items()}
pd_uniques

# Keys for both dictionaries are the same
pd_uniques.keys() == uniques_dataset.keys()

comparing_dicts = {pd_tup[0]:set(pd_tup[1]).symmetric_difference(uni_ds_tup[1]) for pd_tup, uni_ds_tup in zip(pd_uniques.items(), uniques_dataset.items()) if pd_tup[0] == uni_ds_tup[0] }

{k:len(v) for k,v in comparing_dicts.items()}

comparing_dicts["Airline Name"]
comparing_dicts["Claim Type"]
comparing_dicts["Item Category"] # Looks like there are multiple Item categories in this column






"""
### Need to compare the lists for each

# TODO: Make function that helps dedup similar names
# TODO: apply to all things you have to one hot encode
# remove spaces
# make lower case
# compare
# assign new name**
l_temp = [str(name) for name in l1]
l_temp.sort()
l_temp
l2 = {col_name:re.sub(" ", "", str(col_name).lower()) for col_name in l1}
len(l1)
len(set([val for val in l2.values()]))


# remove spaces
# make lower case
# compare
# assign new name
"""

dummies.head()

data_modeling = data_clean.drop(cols_for_dummies, axis = 1)
data_modeling.drop(["Date Received", "Incident D"], axis = 1, inplace = True)
data_modeling  = pd.concat([data_modeling, dummies], axis = 1)
data_modeling.drop(["Date Received", "Incident D"], axis = 1, inplace = True) # Two sets of columns with this name
data_modeling.shape
data_modeling.head()
data_modeling.isnull().sum().sum()

data_modeling.dtypes.values

numb_bad_col_types = sum([True if col_type not in ["uint8", "float64", "int64"] else False for col_type in data_modeling.dtypes])
print("The number of columns that need checking are: ", numb_bad_col_types)
[col_type for col_type in data_modeling.dtypes if col_type not in ["uint8", "float64", "int64"]]


X = data_modeling.drop(["Close Amount", "Date Recieved", "Incident D"], axis = 1)
Y = data_modeling["Close Amount"]
from sklearn import tree

import pydot
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image

os.environ["PATH"] += os.pathsep + "C:/Users/board/Desktop/Kaggle/release/bin"

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


clf = tree.DecisionTreeRegressor(max_depth = 3, max_leaf_nodes=10, min_samples_leaf=10)
clf.fit(X,Y)
clf.score(X,Y)

graphs2, = print_tree(clf, features = X.columns)
Image(graphs2.create_png())

dir(clf)

feat_importances = pd.DataFrame({"Features":X.columns, "Importances":clf.feature_importances_})
feat_importances.sort_values(by="Importances", ascending=False).head(10)
non_zero_importances = feat_importances[feat_importances["Importances"] > 0.0]
non_zero_feats = non_zero_importances["Features"].values
