# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper import plot_decision_boundary
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
sns.set_style('white')


# Read the dataset as a pandas dataframe
df = pd.read_csv("boostingclassifier.csv")

# Read the columns latitude and longitude as the predictor variables
X = df[['latitude','longitude']].values

# Landtype is the response variable
y = df['landtype'].values

### edTest(test_response) ###
# update the class labels to appropriate values for AdaBoost
y = np.where(y == 0, -1, 1)

# AdaBoost algorithm implementation from scratch

def AdaBoost_scratch(X, y, M=10):
    '''
    X: data matrix of predictors
    y: response variable
    M: number of estimators (e.g., 'stumps')
    '''
    N = len(y)
    estimator_list = []
    estimator_weight_list = np.zeros(M)
    sample_weight_list = np.zeros((M+1, N))
    y_predict_list = np.zeros((M, N), dtype=int)
    estimator_error_list = np.zeros(M)

    sample_weight = np.ones(N) / N
    sample_weight_list[0] = sample_weight

    for m in range(M):
        estimator = DecisionTreeClassifier(max_depth=2)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)
        incorrect = (y_predict != y).astype(int)
        estimator_error = np.average(incorrect, weights=sample_weight)
        estimator_weight = 0.8 * 0.5 * np.log((1 - estimator_error) / (estimator_error + 1e-10))
        sample_weight *= np.exp(estimator_weight * incorrect)
        sample_weight /= sample_weight.sum()

        estimator_list.append(estimator)
        y_predict_list[m] = y_predict
        estimator_error_list[m] = estimator_error
        estimator_weight_list[m] = estimator_weight
        sample_weight_list[m+1] = sample_weight

    # Vectorized prediction
    preds = np.sign(np.dot(estimator_weight_list, y_predict_list))

    return np.array(estimator_list), estimator_weight_list, sample_weight_list, preds

### edTest(test_adaboost) ###
# Call the AdaBoost function to perform boosting classification
estimator_list, estimator_weight_list, sample_weight_list, preds  = \
AdaBoost_scratch(X, y, M=50)

# Calculate the model's accuracy from the predictions returned above
accuracy = (np.sum(preds == y)) / len(y)
print(f'accuracy: {accuracy:.3f}')

# Helper code to plot the AdaBoost Decision Boundary stumps
fig = plt.figure(figsize = (16,16))
for m in range(0, 9):
    fig.add_subplot(3,3,m+1)
    s_weights = (sample_weight_list[m,:] / sample_weight_list[m,:].sum() ) * 300
    plot_decision_boundary(estimator_list[m], X,y,N = 50, scatter_weights =s_weights,counter=m)
    plt.tight_layout()
    

# Use sklearn's AdaBoostClassifier to take a look at the final decision boundary 

# Initialise the model with Decision Tree classifier as the base model same as above
# Use SAMME as the algorithm and 9 estimators
boost = AdaBoostClassifier( estimator = DecisionTreeClassifier(max_depth = 1), 
                            algorithm = 'SAMME', n_estimators=9)

# Fit on the entire data
boost.fit(X,y)

# Call the plot_decision_boundary function to plot the decision boundary of the model 
plot_decision_boundary(boost, X,y, N = 50)

plt.title('AdaBoost Decision Boundary', fontsize=16)
plt.show()


### edTest(test_chow1) ###
# Type your answer within in the quotes given
answer1 = 'Less than 1'
