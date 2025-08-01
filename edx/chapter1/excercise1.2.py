# Import necessary libraries
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 20)
plt.rcParams["figure.figsize"] = (12,8)


# Read the datafile "county_election_train.csv" as a Pandas dataframe
elect_train = pd.read_csv("data/county_election_train.csv")

# Read the datafile "county_election_test.csv" as a Pandas dataframe
elect_test = pd.read_csv("data/county_election_test.csv")

# Take a quick look at the dataframe
elect_train.head()

### edTest(test_response) ###
x_train = elect_train[["minority", "bachelor"]]
x_test = elect_test[["minority", "bachelor"]]
# Creating the response variable
elect_train["_estimator"] = elect_train["trump"] > elect_train["clinton"].astype(int)
elect_test["_estimator"] = elect_test["trump"] > elect_test["clinton"].astype(int)
# Set all the rows in the train data where "trump" value is more than "clinton"
# Ensure the results are binary i.e. 0s or 1s
y_train = elect_train["_estimator"]
# Set all the rows in the test data where "trump" value is more than "clinton"
# Ensure the results are binary i.e. 0s or 1
y_test = elect_test["_estimator"]


# Plot "minority" vs "bachelor" from the train data for Trump and Clinton
# Plot minority on the x-axis and bachelor on the y-axis
# Use different colours to depict data points associated with Trump and Clinton

plt.scatter(elect_train[elect_train["_estimator"]==1]["minority"], elect_train[elect_train["_estimator"]==1]["bachelor"],marker=".",color="blue",label="Trump", s=50, alpha=0.4)
plt.scatter(elect_train[elect_train["_estimator"]==0]["minority"], elect_train[elect_train["_estimator"]==0]["bachelor"],marker=".",color="green",label="Clinton", s=50, alpha=0.4)

plt.xlabel("minority")
plt.ylabel("bachelor")
plt.legend()
plt.show();


### edTest(test_model) ###
# Initialize a Decision Tree classifier of depth 3
# Choose Gini as the splitting criteria
dtree = DecisionTreeClassifier(criterion='gini', max_depth=3)
X_train = elect_train[["minority", "bachelor"]]
X_test = elect_test[['minority', 'bachelor']]
# Fit the classifier on the train data
# but only use the minority column as the predictor variable
dtree.fit(X_train, y_train)


# Code to set the size of the plot
plt.figure(figsize=(12,8))

# Plot the Decision Tree trained above with parameters filled as True
tree.plot_tree(
    dtree,
    feature_names=dtree.feature_names_in_,
    class_names=['clinton', 'trump'],
    filled=True,
    precision=0
)

plt.title("Decision Tree for 2016 Presidential Election Prediction")

plt.show();
