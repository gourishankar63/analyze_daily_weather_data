import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Step 1: Load the Data
try:
    df = pd.read_csv('weather.csv')
except FileNotFoundError:
    print("The file 'weather.csv' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print("The file is empty.")
    exit()
except pd.errors.ParserError:
    print("Error parsing the file.")
    exit()

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df = df.dropna()

# Step 3: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# Step 4: Feature Engineering
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])  # Drop rows where date conversion failed
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Additional feature: Temperature range
df['TempRange'] = df['MaxTemp'] - df['MinTemp']

# Step 5: Data Analysis
# Example: Calculate average MaxTemp by month
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

# Step 6: Data Visualization (Part 2)
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.show()

# Step 7: Advanced Analysis (e.g., predict Rainfall)
# Prepare the data for prediction
features = ['MinTemp', 'MaxTemp', 'TempRange']
X = df[features]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Try Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression - Mean Squared Error: {mse_lr}, R2 Score: {r2_lr}')

# Try Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest - Mean Squared Error: {mse_rf}, R2 Score: {r2_rf}')

# Cross-validation for Random Forest
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print(f'Cross-validated MSE for Random Forest: {cv_mse}')

# Step 8: Conclusions and Insights
# Identify the highest and lowest rainfall months
monthly_avg_rainfall = df.groupby('Month')['Rainfall'].mean()
highest_rainfall_month = monthly_avg_rainfall.idxmax()
lowest_rainfall_month = monthly_avg_rainfall.idxmin()
print(f'Highest rainfall month: {highest_rainfall_month}, Lowest rainfall month: {lowest_rainfall_month}')

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_rainfall.index, monthly_avg_rainfall.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Rainfall')
plt.title('Monthly Average Rainfall')
plt.grid(True)
plt.savefig('monthly_avg_rainfall.png')
plt.show()

