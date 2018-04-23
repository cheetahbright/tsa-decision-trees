# Analyze the embdeddings

from importlib import reload
import clean_data2
reload(clean_data2)
from clean_data2 import *

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

embeddings.index.values.tolist()

[x for x in embeddings.index.values.tolist() if "china" in x.lower()]

[x for x in embeddings.index.values.tolist() if "american" in x.lower()]

# Using embeddings (very helpful) https://medium.com/swlh/playing-with-word-vectors-308ab2faa519
# Scipy cosine distance -- https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.cosine.html
# The other link has a version of it too


from scipy.spatial.distance import cosine, euclidean

embeddings.iloc[0,:]

def cosine_sim(a,b):
    """
    Returns cosine similiarity using the cosine distance function from scipy
    """
    cosine_dist = cosine(a,b)
    cosine_sim = 1-cosine_dist
    return(cosine_sim)

def sorted_by_similarity(base, embeddings_df):
    """Returns words sorted by cosine distance to a given vector, most similar first"""
    vectors_with_sim = [(cosine_sim(base, row), name) for name, row in embeddings_df.iterrows()]
    # We want cosine similarity to be as large as possible (close to 1)
    return sorted(vectors_with_sim, key=lambda t: t[0], reverse=True)


master_dict1 = dict()
for name, row in embeddings.iterrows():
    master_dict1[name] = sorted_by_similarity(row, embeddings)

[k for k in master_dict1.keys() if "jet" in k.lower()]


master_dict1["Airline Name_Air New Zealand"][:5]
master_dict1["Airline Name_Air New Zealand"][-5:]


[x for x in airline_col_names if "zealand" in x.lower()]
