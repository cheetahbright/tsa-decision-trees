# Analyze the embdeddings

from importlib import reload
import clean_data
reload(clean_data)
from clean_data import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cleaned_data = clean_data_f()
#data0 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")

# Find airline columns
airline_col_names = [x for x in cleaned_data.columns if "Airline" in x]

airline_df = cleaned_data[airline_col_names]
airline_df_small = airline_df.idxmax(axis = 1)
airline_df_small.rename( "Airline Name", inplace = True)
airline_df_small.head()

len(airline_df.columns) == airline_df_small.nunique()
# Should be True


airline_close_df= pd.concat([cleaned_data["Close Amount"], airline_df_small], axis = 1)
airline_close_df.shape
airline_close_df.columns
airline_close_df.head()
airline_gb = airline_close_df.groupby("Airline Name")

airline_gb.describe()["Close Amount"].sort_values("mean", ascending = False)



#### Get embeddings

embeddings = pd.read_csv("Embeddings_model2.csv")
embeddings.shape
embeddings.head()
embeddings.set_index("AirlineName", inplace = True)
embeddings.head()


# Using embeddings (very helpful) https://medium.com/swlh/playing-with-word-vectors-308ab2faa519
# Scipy cosine distance -- https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.cosine.html
# The other link has a version of it too
