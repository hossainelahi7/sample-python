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
# Creating the response variable
elect_train["_estimator"] = elect_train["trump"] > elect_train["clinton"].astype(int)
elect_test["_estimator"] = elect_test["trump"] > elect_test["clinton"].astype(int)

x_train = elect_train.drop(columns=['_estimator', 'trump', 'clinton'])
x_test = elect_test.drop(columns=['_estimator', 'trump', 'clinton'])
# Set all the rows in the train data where "trump" value is more than "clinton"
# Ensure the results are binary i.e. 0s or 1s
y_train = elect_train["_estimator"]
# Set all the rows in the test data where "trump" value is more than "clinton"
# Ensure the results are binary i.e. 0s or 1
y_test = elect_test["_estimator"]


# Plot "minority" vs "bachelor" from the train data for Trump and Clinton
# Plot minority on the x-axis and bachelor on the y-axis
# Use different colours to depict data points associated with Trump and Clinton
trump_data = elect_train[elect_train["_estimator"] == 1]
clinton_data = elect_train[elect_train["_estimator"] == 0]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(trump_data["minority"], trump_data["bachelor"], color='red', label='Trump', alpha=0.6)
plt.scatter(clinton_data["minority"], clinton_data["bachelor"], color='blue', label='Clinton', alpha=0.6)

# Add labels, title, and legend
plt.xlabel("Minority")
plt.ylabel("Bachelor")
plt.title("Minority vs Bachelor (Train Data)")
plt.legend()
plt.grid(True)
plt.show()


### edTest(test_model) ###
# Initialize a Decision Tree classifier of depth 3
# Choose Gini as the splitting criteria
dtree = DecisionTreeClassifier(criterion='gini', max_depth=3)
# Fit the classifier on the train data
# but only use the minority column as the predictor variable
X_train = elect_train[["minority"]]
dtree.fit(X_train, y_train)


# Code to set the size of the plot
plt.figure(figsize=(12,8))

# Plot the Decision Tree trained above with parameters filled as True
tree.plot_tree(
    dtree,
    feature_names=X_train.columns,
    class_names=['clinton', 'trump'],
    filled=True
)

plt.title("Decision Tree for 2016 Presidential Election Prediction")

plt.show();
