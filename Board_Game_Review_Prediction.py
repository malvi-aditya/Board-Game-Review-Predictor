#Import Required Libraries
import matplotlib.pyplot as plt, pandas as pd,seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


#Load and Print properties of dataset
dataset=pd.read_csv("games.csv")
print("Columns:- ", dataset.columns)
print("Size of data:- ", dataset.shape)
print()
#Histogram
print("Histogram for average rating(to be predicted):- ")
dataset.hist("average_rating")
plt.show()


#First row with average rating 0
print("First row with average rating 0")  
print(dataset[dataset["average_rating"]==0].iloc[0])
print()
#First row with average rating > 0
print("First row with average rating > 0")
print(dataset[dataset["average_rating"]>0].iloc[0]) 


#Remove rows with no reviews
dataset= dataset[dataset["users_rated"]>0] 
dataset=dataset.dropna(axis=0)
#After removing games with no reviews and missing values
print()
print("Histogram after removing games with no reviews and missing values")
dataset.hist("average_rating")
plt.show()


#Correlation matrix and Heatmap
mat=dataset.corr()
fig=plt.figure(figsize=(12,9))
print("Heatmap:  ")
sb.heatmap(mat,vmax= .8,square=True)
plt.show()


#Filter columns not required
col=dataset.columns.tolist()
#Remove unwanted columns,average_rating is the target so removed
col=[i for i in col if i not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]
#target set
target="average_rating"
print(80*"-")


print("Regression: ")
print()
# Generate the training set.  Set random_state to be able to replicate results.
train=dataset.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test=dataset.loc[~dataset.index.isin(train.index)]
# Print the shapes of both sets.
print("Train data size: ",train.shape)
print("Test data size: ",test.shape)


#load the model then fit,LINEAR MODEL
lr=LinearRegression()
lr.fit(train[col], train[target])


#Predict
output=lr.predict(test[col])
print("Error from Linear Regression Model: ",mean_squared_error(output,test[target]))  #error


#NON LINEAR MODEL,Decision tree
# Initialize the model,fit the model to the data.
dt=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
dt.fit(train[col], train[target])


# Predict,error
predictions=dt.predict(test[col])
print("Error from Non-Linear Model(Decision tree): ",mean_squared_error(predictions, test[target]))


