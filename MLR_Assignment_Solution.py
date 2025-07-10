
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("ToyotaCorolla - MLR.csv")

# Rename columns for consistency
df.rename(columns={'Age_08_04': 'Age', 'cc': 'CC', 'Fuel_Type': 'FuelType'}, inplace=True)

# Select relevant columns
df = df[['Price', 'Age', 'KM', 'FuelType', 'HP', 'Automatic', 'CC', 'Doors', 'Weight']]

# EDA: Summary statistics and correlation heatmap
print(df.describe(include='all'))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Encode categorical variable
df_encoded = pd.get_dummies(df, columns=['FuelType'], drop_first=True)

# Split dataset
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: All features
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Model 1 R2:", r2_score(y_test, y_pred))

# Model 2: Drop CC and Doors
X2 = X.drop(['CC', 'Doors'], axis=1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42)
lr2 = LinearRegression()
lr2.fit(X2_train, y2_train)
y2_pred = lr2.predict(X2_test)
print("Model 2 R2:", r2_score(y2_test, y2_pred))

# Model 3: Only Age, KM, Weight
X3 = X[['Age', 'KM', 'Weight']]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.2, random_state=42)
lr3 = LinearRegression()
lr3.fit(X3_train, y3_train)
y3_pred = lr3.predict(X3_test)
print("Model 3 R2:", r2_score(y3_test, y3_pred))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso R2:", r2_score(y_test, lasso_pred))

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge R2:", r2_score(y_test, ridge_pred))
