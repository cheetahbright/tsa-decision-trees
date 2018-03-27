import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import re

os.chdir(r"C:\\Users\\board\\Desktop\\Kaggle\\tsa-decision-trees")
data1 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")

data1.shape
data1.head()
data1.describe()

col_n_unique = data1.nunique()

data2 = data1.replace("-", np.NaN)
col_n_missing = data2.isnull().sum()

col_stats = pd.concat([col_n_missing, col_n_unique], axis =1)
col_stats.columns = ["N_Missing", "N_Unique"]
col_stats

# Begin looking at CLose Amount
y = data2["Close Amount"]
y.describe()
len(y[y ==0])/len(y)
y.hist(bins = 20)
y[y!=0].hist(bins = 20)

# TODO: Dates analysis
data_clean = data2[pd.notnull(data2["Close Amount"])]
cols_for_dummies = ['Airport Code', 'Airport Name', 'Airline Name', 'Claim Type', 'Claim Site', 'Item Category', 'Disposition']
dummies = pd.get_dummies(data_clean, columns = cols_for_dummies)
dummies.head()
data_clean.head()


l1 = data2["Airline Name"].unique()

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



data_clean.head()
X = data_clean.drop(["Close Amount"], axis = 1)
Y = data_clean["Close Amount"]
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf.fit(X,Y)
