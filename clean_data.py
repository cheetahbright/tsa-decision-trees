import pandas as pd
import numpy as np
import logging
logging.basicConfig(filename='clean_data.log',level=logging.INFO)

def clean_data_f():

    try:
        data0 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")
        logging.debug("Shape of original data is"+str(data0.shape))
    except:
        print("Could not read data into pandas df")
        return()
    # Validations: check columns have names used in this file
    # Logger for each step... to file to show variables were created


    data1 = data0.replace("-", np.NaN)
    data2 = data1[pd.notnull(data1["Close Amount"])]

    print("Create time series variables...")
    #data2["Date Received"] = pd.DatetimeIndex(data2["Date Received"])
    #data2["Incident D"] = pd.DatetimeIndex(data2["Incident D"])
    date_rec_dti = pd.DatetimeIndex(data2["Date Received"])
    incident_date_dti = pd.DatetimeIndex(data2["Incident D"])

    date_rec_dow_hot = pd.get_dummies(date_rec_dti.dayofweek, prefix = "DateRecDOW")
    inc_date_dow_hot = pd.get_dummies(incident_date_dti.dayofweek, prefix = "IncidentDOW")

    open_days = (date_rec_dti - incident_date_dti).days
    logging.info("Finished creating time series variables")


    print("Create dummy variables...")
    cols_for_dummies = ['Airport Code', 'Airport Name', 'Airline Name',
    'Claim Type', 'Claim Site', 'Item Category', 'Disposition']
    # One hot encoding
    dummies = pd.get_dummies(data2[cols_for_dummies])
    dict1 = dict()
    for col_base in cols_for_dummies:
        dict1[col_base] = sum([])

    data_modeling = data2.drop(cols_for_dummies, axis = 1)
    data_modeling.drop(["Date Received", "Incident D"]
    , axis = 1, inplace = True)
    data_modeling  = pd.concat([data_modeling, dummies], axis = 1)
    log_message = "Created one hot variables " + str(dummies.columns)
    logging.info(log_message)

    data_modeling.shape

    print("Conducting validations... ")

    numb_bad_col_types = sum([True if col_type not in
    ["uint8", "float64", "int64"] else False
    for col_type in data_modeling.dtypes])

    data_modeling.set_index("Claim Number")
    print("Success returning modeling dataset")
    print("Dataset is of shape: ", data_modeling.shape)
    return(data_modeling)
