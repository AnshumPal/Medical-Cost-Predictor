# importing the data from the website
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mirichoi0218/insurance")

print("Path to dataset files:", path)

#reading the data
import pandas as pd

df = pd.read_csv("insurance.csv")
print(df.head())
print(df.shape)

# checking if there are any null value in the dataset
df.isnull()

df.isnull().sum()

# it will drop all the rows which contain atleast 1 missing value
df.dropna(axis=0)

# it will drop all the coloumns which contain atleast 1 missing value
df.dropna(axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from  sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# converting the data
df['age'] = df['age'].astype('float64')
df['bmi'] = df['bmi'].astype('float64')
df['children'] = df['children'].astype('int64')
df['charges'] = df['charges'].astype('float64')


# handling the Categorical variable (One-Hot-Coding)
categorical_columns = ['sex', 'smoker', 'region']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)


# We are scaling the numeric columns (age, bmi, children, charges) to have a mean of 0 and a standard deviation of 1, making the data more suitable for machine learning models.
Scaler = StandardScaler()
numeric_columns = ['age', 'bmi', 'children']
df[numeric_columns] = Scaler.fit_transform(df[numeric_columns])


# assuming charge is the target value and dropping it
X = df.drop('charges', axis=1)
y = df['charges']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)



 # Model training
 from sklearn.ensemble import RandomForestRegressor

 model = RandomForestRegressor(random_state=56)
 model.fit(X_train, y_train)

# model evaluation


from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Hyperparameter Tuning with GridSearchCV for the better performance of the model which try different combination and train the model for the better performance
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Split data into X and y
X = df.drop('charges', axis=1)
y = df['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessor (standard scaling for numerical data, one-hot encoding for categorical)
numerical_columns = ['age', 'bmi', 'children']
categorical_columns = ['sex_male', 'smoker_yes', 'region_southwest', 'region_northwest', 'region_southeast']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ]
)

# Define the model inside a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter grid for tuning
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best model and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test Score of Best Model:", test_score)


# cross validation :divide the data into small chunks and train the data and test it and then again repeat the process testing with a different chunks and training with other
from sklearn.model_selection import cross_val_score

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')

# Mean cross-validation score
print("Cross-validation MSE scores:", -cv_scores)
print("Mean CV MSE:", -cv_scores.mean())


# saving the model for future use
import joblib

# Assuming you've run GridSearchCV and stored the best model in `grid_search`
best_rf = grid_search.best_estimator_

# Save the model
joblib.dump(best_rf, 'best_random_forest_model.pkl')

# Load the model
loaded_model = joblib.load('best_random_forest_model.pkl')

# Use the model to make predictions
loaded_predictions = loaded_model.predict(X_test)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. **Data Overview**: Getting a quick glance at the data
print("Data Overview:")
print(df.head())  # View the first few rows

# 2. **Summary Statistics**: Checking summary statistics of numeric columns
print("\nSummary Statistics:")
print(df.describe())  # Statistical summary

# 3. **Data Visualizations**:

## a. **Histograms**: To visualize the distribution of numeric columns
df[['age', 'bmi', 'children', 'charges']].hist(bins=15, figsize=(12, 8))
plt.suptitle('Histograms of Numeric Columns', y=1.02)
plt.show()

## b. **Pair Plot**: To visualize the relationships between features
sns.pairplot(df[['age', 'bmi', 'children', 'charges']])
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.show()

## c. **Correlation Heatmap**: To check the correlation between features
corr_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 4. **Model Performance Visualizations**:

## a. **Regression Metrics**: Evaluate model performance using regression metrics
y_pred = best_rf.predict(X_test)

# Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Extract the trained RandomForest model from the pipeline
model = best_rf.named_steps['regressor']  # Changed from 'randomforestclassifier' to 'regressor'

# Now, get the feature importances
importances = model.feature_importances_
# Now, get the feature importances
importances = model.feature_importances_
features = X_train.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot Feature Importances
plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=features[indices], palette='viridis')
plt.title('Feature Importances')
plt.show()

# Saving the Model (optional)
import joblib
joblib.dump(best_rf, 'best_random_forest_model.pkl')  # Save the model




