# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline

# Introduction and Caveat
# This notebook focuses on various feature ranking methods including RFE, Stability Selection, Linear Models, and Random Forest.
# Note: Execution might take a few minutes.

# 1. Data Cleaning and Visualization

# Reading the dataset
data_path = '/path/to/dataset.csv'  # Update the path to your dataset
data = pd.read_csv(data_path)
data.head()

# Removing unnecessary columns
data.drop(['Unnamed: 0', 'PASSENGER_ID'], axis=1, inplace=True)

# Checking for null values and data types
print("Checking for null values:\n", data.isnull().any())
print("\nData types:\n", data.dtypes)

# Pairplot Visualization
# Exploring relationships between features
pairplot_columns = ['PAX_LOYALTY_FLAG', 'PAX_AGE', 'NET_TICKET_REVENUE_USD', 'PAX_IS_DRINKING_AGE', 'PAX_IS_KID']
sns.pairplot(data[pairplot_columns], size=6)
plt.show()

# Correlation Heatmap
# Identifying correlations between features
correlation_matrix = data.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="cubehelix", linewidths=0.25)
plt.title('Pearson Correlation of Features')
plt.show()

# 2. Stability Selection via Randomized Lasso

# Extracting target variable and feature matrix
target = 'TARGET_VARIABLE'  # Replace with your target column name
X = data.drop(target, axis=1)
Y = data[target]

# Feature Ranking using Randomized Lasso
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)

# Utility function for ranking features
def rank_features(scores, feature_names):
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    ranks = dict(zip(feature_names, scores))
    return ranks

ranks = {}
ranks["Stability"] = rank_features(np.abs(rlasso.scores_), X.columns)

# 3. Recursive Feature Elimination (RFE)

# Applying RFE using Linear Regression
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X, Y)
ranks["RFE"] = rank_features(rfe.ranking_, X.columns, order=-1)

# 4. Linear Model Feature Ranking

# Applying Linear, Ridge, and Lasso Regression
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=7),
    "Lasso": Lasso(alpha=0.05)
}

for name, model in models.items():
    model.fit(X, Y)
    ranks[name] = rank_features(np.abs(model.coef_), X.columns)

# 5. Random Forest Feature Ranking

# Feature ranking with Random Forest
rf = RandomForestRegressor(n_estimators=50)
rf.fit(X, Y)
ranks["Random Forest"] = rank_features(rf.feature_importances_, X.columns)

# 6. Creating the Feature Ranking Matrix

# Combining and averaging feature rankings
feature_ranks = pd.DataFrame(ranks)
feature_ranks['Mean'] = feature_ranks.mean(axis=1)
feature_ranks = feature_ranks.sort_values(by='Mean', ascending=False)

# Plotting feature importances
sns.barplot(x="Mean", y=feature_ranks.index, data=feature_ranks)
plt.title('Feature Importances')
plt.show()
