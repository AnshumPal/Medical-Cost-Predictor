import kaggle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import xgboost as xgb


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Download dataset using Kaggle API
kaggle.api.dataset_download_files('mirichoi0218/insurance', path='.', unzip=True)
print("Dataset downloaded successfully.")

# Reading the data
df = pd.read_csv("insurance.csv")
print(df.head())
print(df.shape)

# Checking for null values
print(df.isnull().sum())

# Drop rows/columns with missing values (just shown, not used further)
df.dropna(axis=0)
df.dropna(axis=1)

# Converting the data types
df['age'] = df['age'].astype('float64')
df['bmi'] = df['bmi'].astype('float64')
df['children'] = df['children'].astype('int64')
df['charges'] = df['charges'].astype('float64')

# One-hot encoding categorical columns
categorical_columns = ['sex', 'smoker', 'region']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Scaling numeric columns
Scaler = StandardScaler()
numeric_columns = ['age', 'bmi', 'children']
df[numeric_columns] = Scaler.fit_transform(df[numeric_columns])

# Splitting into X and y
X = df.drop('charges', axis=1)
y = df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train individual models
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, objective='reg:squarederror')

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Get predictions from each model
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Blend predictions using weighted averaging
blend_ratio_rf = 0.6  # More weight on RF for stability
blend_ratio_xgb = 0.4  # XGBoost for efficiency
y_pred_blend = (blend_ratio_rf * y_pred_rf) + (blend_ratio_xgb * y_pred_xgb)

# Evaluate blended model
mse_blend = mean_squared_error(y_test, y_pred_blend)
r2_blend = r2_score(y_test, y_pred_blend)
print(f"Blended Model - Mean Squared Error: {mse_blend}")
print(f"Blended Model - R-squared: {r2_blend}")

# Saving blended predictions
joblib.dump((rf_model, xgb_model, blend_ratio_rf, blend_ratio_xgb), 'blended_model.pkl')


# Hyperparameter tuning
numerical_columns = ['age', 'bmi', 'children']
categorical_columns = ['sex_male', 'smoker_yes', 'region_southwest', 'region_northwest', 'region_southeast']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test Score of Best Model:", test_score)

# Cross-validation
rf = RandomForestRegressor(random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
print("Cross-validation MSE scores:", -cv_scores)
print("Mean CV MSE:", -cv_scores.mean())

# Saving the model
joblib.dump(best_model, 'best_random_forest_model.pkl')

# Loading and using the model
loaded_model = joblib.load('best_random_forest_model.pkl')
loaded_predictions = loaded_model.predict(X_test)

# Data Overview
print("Data Overview:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Histograms
df[['age', 'bmi', 'children', 'charges']].hist(bins=15, figsize=(12, 8))
plt.suptitle('Histograms of Numeric Columns', y=1.02)
plt.show()

# Pair Plot
sns.pairplot(df[['age', 'bmi', 'children', 'charges']])
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.show()

# Correlation Heatmap
corr_matrix = df[['age', 'bmi', 'children', 'charges']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Final Model Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Feature Importances
model_rf = best_model.named_steps['regressor']
importances = model_rf.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=features[indices], palette='viridis')
plt.title('Feature Importances')
plt.show()
