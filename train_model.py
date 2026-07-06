# ==========================================
# STEP 1: Import Required Libraries
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import joblib


# ==========================================
# STEP 2: Load the Dataset
# ==========================================

# Read the Excel dataset
df = pd.read_excel("flood dataset.xlsx")

print("Dataset Loaded Successfully!\n")


# ==========================================
# STEP 3: Explore the Dataset
# ==========================================

# Display the first 5 rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Display the dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())
# ==========================================
# STEP 4: Check Missing Values
# ==========================================

print("\nChecking Missing Values:")
print(df.isnull().sum())

print("\nAny Missing Values?")
print(df.isnull().any())
# ==========================================
# STEP 5: Outlier Detection using Box Plot
# ==========================================

# Display box plots for all numerical columns
plt.figure(figsize=(15,8))
df.boxplot()
plt.title("Box Plot for Outlier Detection")
plt.xticks(rotation=45)
plt.show()
# ==========================================
# STEP 5.1: Detect Outliers using IQR
# ==========================================

# Select numerical columns except the target column
columns = df.columns[:-1]

for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"{column}: {len(outliers)} outliers")
    # ==========================================
# STEP 5.2: Handle Outliers using IQR Capping
# ==========================================

for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[column] = np.where(df[column] > upper, upper, df[column])
    df[column] = np.where(df[column] < lower, lower, df[column])

print("\nOutliers have been handled successfully using IQR Capping.")
# ==========================================
# STEP 5.3: Verify After Capping
# ==========================================

plt.figure(figsize=(15,8))
df.boxplot()
plt.title("Box Plot After Outlier Handling")
plt.xticks(rotation=45)
plt.show()
# ==========================================
# STEP 6: Split Dataset into X and y
# ==========================================

# Independent Variables (Input Features)
X = df.drop("flood", axis=1)

# Target Variable (Output)
y = df["flood"]

print("\nIndependent Variables (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)
# ==========================================
# STEP 7: Train-Test Split
# ==========================================

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
# ==========================================
# STEP 8: Feature Scaling
# ==========================================

# Create StandardScaler object
scaler = StandardScaler()

# Fit the scaler on training data and transform it
X_train = scaler.fit_transform(X_train)

# Transform the testing data
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully!")

print("\nScaled Training Data Shape:", X_train.shape)
print("Scaled Testing Data Shape:", X_test.shape)
# ==========================================
# STEP 9: Decision Tree Model
# ==========================================

# Create Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Predict on test data
dt_predictions = dt_model.predict(X_test)

# Evaluate the model
print("\n========== Decision Tree ==========")

accuracy = accuracy_score(y_test, dt_predictions)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, dt_predictions))

print("\nClassification Report:")
print(classification_report(y_test, dt_predictions))
# ==========================================
# STEP 10: Random Forest Model
# ==========================================

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
print("\n========== Random Forest ==========")

accuracy = accuracy_score(y_test, rf_predictions)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))
# ==========================================
# STEP 11: K-Nearest Neighbors (KNN)
# ==========================================

# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Predict on test data
knn_predictions = knn_model.predict(X_test)

# Evaluate the model
print("\n========== KNN ==========")

accuracy = accuracy_score(y_test, knn_predictions)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))

print("\nClassification Report:")
print(classification_report(y_test, knn_predictions))
# ==========================================
# STEP 12: XGBoost Model
# ==========================================

# Create XGBoost model
xgb_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on test data
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the model
print("\n========== XGBoost ==========")

accuracy = accuracy_score(y_test, xgb_predictions)
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_predictions))

print("\nClassification Report:")
print(classification_report(y_test, xgb_predictions))
# ==========================================
# STEP 13: Compare All Models
# ==========================================

dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)

print("\n========== Model Comparison ==========")

print(f"Decision Tree Accuracy : {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"KNN Accuracy          : {knn_accuracy:.4f}")
print(f"XGBoost Accuracy      : {xgb_accuracy:.4f}")
# ==========================================
# STEP 14: Save Model and Scaler
# ==========================================

joblib.dump(xgb_model, "floods.save")
joblib.dump(scaler, "transform.save")

print("\nModel saved as floods.save")
print("Scaler saved as transform.save")