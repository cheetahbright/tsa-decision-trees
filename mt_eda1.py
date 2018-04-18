import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%matplotlib inline
import re

# https://www.dhs.gov/tsa-claims-data
os.getcwd()
if input('Is this m or d?') == 'm':
    os.chdir(r"C:\\Users\\board\\Desktop\\Kaggle\\tsa-decision-trees")
else:
    os.chdir(r'C:\Users\drose\Documents\GitHub\tsa-decision-trees')
data1 = pd.read_excel("claims-data-2015-as-of-feb-9-2016.xlsx")

### start data inspection ###
data1.shape
data1.head(10)
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

#########################
#########################
### Quick time Series analysis
#########################
#########################


# TODO: Dates analysis
data2.head()
data2.columns
data2["Date Received"], data2["Incident D"] = pd.DatetimeIndex(data2["Date Received"]), pd.DatetimeIndex(data2["Incident D"])

data2["Date Received"].describe()
data2["Incident D"].describe()

sorted_DR = data2.sort_values(by="Date Received")
sorted_DR = sorted_DR.set_index("Date Received")
plt.plot(sorted_DR.index, sorted_DR["Close Amount"])

plt.show()
# grouping data by months
monthly_group = sorted_DR.groupby(pd.Grouper(freq = 'M'))
monthly_group["Close Amount"].describe()


# Close amount average by month
plt.title("Average Monthly Close Amount by Date Received Month")
plt.xlabel("Month (Date Received)")
plt.ylabel("Average Close Amount")
plt.plot(monthly_group["Close Amount"].describe()["mean"])
plt.show()

numb_0_by_month = monthly_group["Close Amount"].apply(lambda x: len(x[x==0]))
numb_cnt_by_month = monthly_group["Close Amount"].count()

plt.title("Count of 0 and Number of Observations by Date Received Month")
plt.xlabel("Month (Date Received)")
plt.ylabel("Number of 0's")
plt.legend()
plt.plot(numb_0_by_month.index, numb_0_by_month, 'r--',  numb_cnt_by_month.index, numb_cnt_by_month, 'g--')
plt.show()


# Combine these plz
# Number of 0's per month

numb_0_line, = plt.plot(numb_0_by_month.index, numb_0_by_month, 'r--', label = "Number of 0's")
numb_cnt_line, = plt.plot(numb_cnt_by_month.index, numb_cnt_by_month, 'g--', label = "Total Count")
plt.legend(handles =[numb_0_line, numb_cnt_line])
plt.title("Number of 0s and Total Count vs Month")
plt.show()

# Empiricial check of the percentage
pct_0_month= (numb_0_by_month/numb_cnt_by_month).describe()

####################################
###### Group by analysis ###########
####################################

data2.columns
airport_name_gb = data2.groupby("Airport Name")
airport_name_gb["Close Amount"].apply(np.max).sort_values(ascending = False)
airport_name_gb["Close Amount"].describe().sort_values(by="mean", ascending=False)
# ? is this needed??? (how do averages work again... )
airport_name_gb["Close Amount"].describe()["count"].mean()

def gb_stats(col_name):
    if col_name not in data2.columns:
        print("Col name not in data2 columns: Returning empty dict")
        return({col_name+"_empty":""})
    temp_gb = data2.groupby(col_name)
    descr_gb = temp_gb["Close Amount"].describe().sort_values(by = "mean", ascending = False)
    average_cnt = temp_gb["Close Amount"].describe()["count"].mean()
    #return_dict = {col_name+"gb":temp_gb, col_name+"describe_gb":descr_gb, col_name+"avg_cnt":average_cnt}
    return_dict = {"gb":temp_gb, "describe_gb":descr_gb, "avg_cnt":average_cnt}

    return(return_dict)

temp1 = gb_stats("Claim Type")


gb_cols = ["Airport Name", "Airport Code", "Airline Name", "Claim Type", "Claim Site", "Disposition"]
group_by_dicts = {col:gb_stats(col) for col in gb_cols}

group_by_dicts.keys()
group_by_dicts["Claim Type"].keys()
group_by_dicts["Claim Type"]['avg_cnt']
group_by_dicts["Claim Type"]["describe_gb"]
group_by_dicts["Claim Type"]["describe_gb"].shape


#airport_name_gb["Close Amount"].apply(np.mean).sort_values(ascending = False)

# Another custom way to do this
def custom_stats(close_amount):
    # Utilizes the skipna feature
    std_temp = close_amount.std()
    mean_temp = close_amount.mean()
    q50 = close_amount.quantile(0.50)
    final_series = pd.Series([mean_temp, std_temp, q50], index = ["mean", "std", "quantile50"])
    return(final_series)

# Need to unpack this Series
airport_name_gb["Close Amount"].apply(custom_stats)

#################################################
#################################################
########## DUMMY VARIABLES ANALYSIS #############
#################################################
#################################################


### Begin One hot encoding process
# Start by getting an idea of how many variables we may be adding
data_clean = data2[pd.notnull(data2["Close Amount"])]
cols_for_dummies = ['Airport Code', 'Airport Name', 'Airline Name', 'Claim Type', 'Claim Site', 'Item Category', 'Disposition']



data2[cols_for_dummies].nunique().sum()


# One hot encoding
dummies = pd.get_dummies(data_clean[cols_for_dummies])
dummies.shape
pd.Series(dummies.columns.unique())
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

comparing_dicts['Disposition']
comparing_dicts["Airline Name"]
comparing_dicts["Claim Type"]
testMismatch1 = list(comparing_dicts["Item Category"])[1]
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
