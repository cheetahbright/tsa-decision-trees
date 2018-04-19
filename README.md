# tsa-decision-trees

# Goal 
The goal of this analysis is to understand the amount the TSA pays for different claims. The data contains information about the day the claim was recieved, the incident date, airport name, airline name, claim type, claim site, item category, close amount, and disposition. 

The category we are trying to predict is the close amount, which, we believe is the amount the TSA paid to close the claim. 


# Files: 

## clean_data.py 

This file contains the function do some of the data cleaning and returns a pandas dataframe ready for modeling.
To prepare the data, we clean replace all the "-" in the data with numpy's NaN. 

Next we turn the date recieved and the incident date into date time objects to extract information about the variables. We onehot encode the day of the week and find the number of days between the incident date and the date recieved and call this the number of open days. 

Next we one hot encode the categorical variables -- ['Airport Code', 'Airport Name', 'Airline Name',
'Claim Type', 'Claim Site', 'Item Category', 'Disposition'] 


Finally we drop the date time objects, and combine the date variables with the one hot encoded variables. 


## mt_eda1.py 

This file contains various simple analysese of the data. We look at the number of missing and unique values per column. Some variables are easily identified for one hot encoding. If a variable has continous numbers but not many unique values, we may want to consider one-hot encoding the variable. 

We also consider the number of missing values. If there are too many missing values, we may want to drop the variable. 

In this case, most of the variables must be one hot encoded but it is still important to understand how many variables we may be creating for each variable.

We see claim number has the exact number of unique values as the number of observations. This indicates that claim number is a good candidate for the unique ID. 


We also look at the histogram for the dependent variable "Close Amount"
It is heavily skewed to the left due to the 0s in the dataset. We later see that about 40% of the values are 0. 

Next we examine the date recieved by month and the close amount. We see a decreasing trend in the number of claims per months
as the year progresses. We see a similar trend for the average claim amount. 

There are a lot of groupby analysis one might want to conduct on the data according to each category. I decided to make a dictionary of groupby's according to each column name. This can be useful later on for verify and identifying different trends. 


Finally I discover an inconsistency with the pd get_dummies and number of unique values in each column. The number of columns created by the pd.get_dummies should be the same as the total number of unique values in each of the columns. It was not... 
There are some issues because there is a space after the word and the space gets removed when creating the dummy variables. 














