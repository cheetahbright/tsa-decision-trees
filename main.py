import numpy as np
import os
import pandas as pd
from pprint import pprint as pp
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
data = pd.read_excel('claims-data-2015-as-of-feb-9-2016.xlsx')
data['Airline Name'].dropna().unique()

data.replace('-', np.nan, inplace=True)

data = data[pd.notnull(data['Close Amount'])]
data.head()

X = data.iloc[:, 3:8]
X.head()
y = data.iloc[:,9]
y.head()
X = pd.get_dummies(X)
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
#clf_gini.predict([[4, 4, 3, 3]])
y_pred = clf_gini.predict(X_test)
print(y_pred[0])
