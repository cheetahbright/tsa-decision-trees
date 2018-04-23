# TSNE


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from importlib import reload
import clean_data
reload(clean_data)
from clean_data import *

cleaned_data = clean_data_f()



embeddings = pd.read_csv("Embeddings_model2.csv")
embeddings.head()
embeddings.set_index("AirlineName", inplace = True)


# Create airlin Group by
airline_col_names = [x for x in cleaned_data.columns if "Airline" in x]
airline_df = cleaned_data[airline_col_names]
airline_df_small = airline_df.idxmax(axis = 1)
airline_df_small = airline_df_small.rename( "Airline Name")
airline_close_df= pd.concat([cleaned_data["Close Amount"], airline_df_small], axis = 1)

airline_gb_mean.shape
airline_gb_mean = airline_close_df.groupby("Airline Name")["Close Amount"].mean()
deciles = pd.qcut(airline_gb_mean, 14, labels = False, duplicates = "drop")
airline_gb_mean2 = pd.concat([airline_gb_mean, deciles] , axis = 1)
airline_gb_mean2.columns = ["Close Amount", "Deciles"]
airline_gb_mean2


embeddings.sum(axis = 1).describe()
# Embeddings are on the same scale so no need to normalize/standardize
fit_reduced_embeddings = TSNE(n_components=2, random_state=1).fit(embeddings)
reduced_embeddings = fit_reduced_embeddings.fit_transform(embeddings)
reduced_embeddings.shape

embed_pd = pd.DataFrame(reduced_embeddings, index = embeddings.index)


plt.scatter(embed_pd.iloc[:,0], embed_pd.iloc[:,1], c=airline_gb_mean2["Deciles"])
