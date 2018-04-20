

from importlib import reload
import clean_data
reload(clean_data)
from clean_data import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cleaned_data = clean_data_f()
data0 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")

col_n_unique = data0.nunique()
data2 = data0.replace("-", np.NaN)
col_n_missing = data2.isnull().sum()
col_stats = pd.concat([col_n_missing, col_n_unique], axis =1)
col_stats.columns = ["N_Missing", "N_Unique"]
col_stats.shape
col_stats

# Gather variables for Airline Name
airline_col_names = [x for x in cleaned_data.columns if "Airline" in x]
airline_col_names[:3]
len(airline_col_names)

airline_df = cleaned_data[airline_col_names]
airline_df.shape
Y = cleaned_data["Close Amount"]
# Mine was (should be) (6601, 111)

import keras
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.models import Sequential

n_unique_col = len(airline_col_names)
output_embeddings = min(50, n_unique_col//2)
vocab_size = n_unique_col # In this case?



def build_1h_emb_model(hidden_layer_n = 10 ,trace = 1):

    model = Sequential()
    model.add(Embedding(vocab_size, output_embeddings, input_length = n_unique_col))
    model.add(Flatten())
    model.add(Dense(hidden_layer_n, activation = "relu"))
    model.add(Dense(1, activation = "linear"))

    model.compile(optimizer = "adam", loss = "mse")

    if trace:
        print(model.summary())

    return(model)

def build_1h_drop_batch_emb_model(hidden_layer_n = 10, dropout_p = 0.25, trace = 1):

    model = Sequential()
    model.add(Embedding(vocab_size, output_embeddings, input_length = n_unique_col))
    model.add(Dropout(dropout_p, seed=1234))
    model.add(Flatten())
    model.add(Dense(hidden_layer_n, activation = "relu"))
    model.add(Dropout(dropout_p, seed=123))
    model.add(BatchNormalization())
    model.add(Dense(1, activation = "linear"))

    model.compile(optimizer = "adam", loss = "mse")

    if trace:
        print(model.summary())

    return(model)



def plot_nn_metric(history):
    if type(history) != dict:
        print("History input not dictionary...")
        print("Ending function")
        return()

    n_keys = len(list(history.keys()))
    keyss = list(history.keys())

    if n_keys > 2:
        print("More than 2 objects in model history")
        return(0)
    plt.ioff()

    plt.plot(history[keyss[0]])
    plt.title(keyss[0] + ' vs epoch')
    plt.ylabel(keyss[0])
    plt.xlabel('epoch')

    if n_keys > 1:
        plt.plot(history[keyss[1]])
        plt.legend([keyss[0], keyss[1]], loc='upper left')

    plt.show()
    print("Finished plotting")
    return()



model1 = build_1h_emb_model()
hist = model1.fit(airline_df, Y, validation_split = 0.3, epochs = 3)

# Model evaluation
keyys = hist.history.keys()
keys_no_val = [x if x[:4] != "val_" else x[4:] for x in keyys]
unique_keys = list(set(keys_no_val))

for unique_key in unique_keys:
    temp_hist = {k:v for k,v in hist.history.items() if unique_key in k}
    plot_nn_metric(temp_hist)

## Model 2

model2 = build_1h_drop_batch_emb_model()
hist2 = model2.fit(airline_df, Y, validation_split = 0.3, epochs = 20)


keyys = hist2.history.keys()
keys_no_val = [x if x[:4] != "val_" else x[4:] for x in keyys]
unique_keys = list(set(keys_no_val))

for unique_key in unique_keys:
    temp_hist = {k:v for k,v in hist2.history.items() if unique_key in k}
    plot_nn_metric(temp_hist)


# Retrieve the embeddings
# Expect the matrix to have number of rows = to # of classes
# Number of columns to be number selected above (output embeddings)
model2.layers[0].get_weights()[0]
embed_matrix = pd.DataFrame(model2.layers[0].get_weights()[0])
embed_matrix.index = airline_col_names
embed_matrix.shape

# Peak at top corner -- doesn't give much info
embed_matrix.iloc[1:5, 1:5]

embed_matrix.to_csv("Embeddings_model2.csv", index_label = "AirlineName")
