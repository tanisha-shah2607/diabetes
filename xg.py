import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset

df = pd.read_csv("diabetes.csv")

# Replace zero values in key medical features with their median values
features_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for feature in features_to_replace:
    median_value = df[feature].median()
    df[feature] = df[feature].replace(0, median_value)

# Split dataset into features and target variable
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost Classifier
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)