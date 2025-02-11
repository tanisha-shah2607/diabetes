import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Load dataset
file_path = "./diabetes.csv"  # Adjust path as needed
df = pd.read_csv(file_path)

# Check for null values and fill them with the mean
df.fillna(df.mean(), inplace=True)

# Replace zero values in key medical features with their median values
features_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for feature in features_to_replace:
    median_value = df[feature].median()
    df[feature] = df[feature].replace(0, median_value)

# Remove outliers using Z-score threshold of 3
df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# Split dataset into features and target variable
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression for feature importance
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)

# Get feature importance from Logistic Regression
feature_importance = np.abs(logreg.coef_[0])
important_features = X.columns[feature_importance > np.percentile(feature_importance, 20)]  # Keep top 80% important features
X_train_cleaned = X_train[important_features]
X_test_cleaned = X_test[important_features]

# Train XGBoost on cleaned data
model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train_cleaned, y_train)

# Make predictions
y_pred = model.predict(X_test_cleaned)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy after handling null values: {accuracy:.2f}")
print("Classification Report:\n", report)