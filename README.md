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







