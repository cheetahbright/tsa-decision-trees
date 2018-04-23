import pandas as pd
import numpy as np
import logging
logging.basicConfig(filename='clean_data2.log',level=logging.INFO)

def clean_data_f():

    try:
        data0 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")
        logging.info("Shape of original data is"+str(data0.shape))
    except:
        print("Could not read data into pandas df")
        return()
    # Validations: check columns have names used in this file
    # Logger for each step... to file to show variables were created


    data1 = data0.replace("-", np.NaN)
    data2 = data1[pd.notnull(data1["Close Amount"])]

    print("Create time series variables...")
    data2["Date Received"] = pd.DatetimeIndex(data2["Date Received"])
    data2["Incident D"] = pd.DatetimeIndex(data2["Incident D"])
    data2.loc[:,"Date Received DOW"] = pd.DatetimeIndex(data2["Date Received"]).dayofweek
    data2.loc[:,"Incident D DOW"] = pd.DatetimeIndex(data2["Incident D"]).dayofweek
    [data2[x].dtype for x in data2.columns if "incident" in x.lower()]
    #date_rec_dow_hot = pd.get_dummies(date_rec_dti.dayofweek, prefix = "DateRecDOW")
    #inc_date_dow_hot = pd.get_dummies(incident_date_dti.dayofweek, prefix = "IncidentDOW")

    #time_onehot_vars = date_rec_dow_hot.columns.append(inc_date_dow_hot.columns)

    open_days = pd.DatetimeIndex(data2["Date Received"] - data2["Incident D"])
    data2.loc[:, "Open_Days"] = open_days
    logging.info("Finished creating time series variables")


    print("Create dummy variables...")
    dummy_cols_from_df = ['Airport Name', 'Airline Name',
    'Claim Type', 'Claim Site', 'Item Category', 'Disposition']
    # Removed airport code because it is a mapping to airport name
    dummy_cols_from_df.append("Date Received DOW")
    dummy_cols_from_df.append("Incident D DOW")
    # Logic for flooring dummy variables according to count
    def flooring_dummy_vars(col_name, min_obs = 10):
        gb_count = data2.groupby(col_name)["Close Amount"].count()
        indexes_to_replace = set([index for index, value in gb_count.iteritems() if value < min_obs])
        simplified_col = data2[col_name].apply(lambda x: x if x not in indexes_to_replace else "Lessthan"+str(min_obs)+"obs")
        data2[col_name] = simplified_col
        n_col_removed = len(indexes_to_replace)

        log_msg1 = "For col name:"+col_name+" the number of columns removed was "+str(n_col_removed)
        log_msg2 = "They are ... "
        logging.info(log_msg1+"\n"+log_msg2)
        logging.info(indexes_to_replace)

        print(log_msg1)

        return()

    for col in dummy_cols_from_df:
        flooring_dummy_vars(col)


    # One hot encoding
    dummies = pd.get_dummies(data2[dummy_cols_from_df])
    dict1 = dict()
    for col_base in dummy_cols_from_df:
        dict1[col_base] = sum([])

    data_modeling = data2.drop(dummy_cols_from_df, axis = 1)
    data_modeling.drop(["Date Received", "Incident D"]
    , axis = 1, inplace = True)
    data_modeling  = pd.concat([data_modeling, dummies], axis = 1)
    log_message = "Created one hot variables "
    logging.info(log_message)
    logging.info(dummies.columns)


    print("Conducting validations... ")

    numb_bad_col_types = sum([True if col_type not in
    ["uint8", "float64", "int64"] else False
    for col_type in data_modeling.dtypes])

    data_modeling.set_index("Claim Number")
    print("Success returning modeling dataset")
    print("Dataset is of shape: ", data_modeling.shape)
    return(data_modeling)
