import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("HOSPITALS/hospital1", exist_ok=True)
os.makedirs("HOSPITALS/hospital2", exist_ok=True)
os.makedirs("CENTRAL_SERVER/Models", exist_ok=True)
os.makedirs("CENTRAL_SERVER/Storage", exist_ok=True)

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")
target = "diabetes"

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=target)
y = df[target]

# Stratified split into 3 sets
X_temp, X_set3, y_temp, y_set3 = train_test_split(
    X, y, test_size=0.248, stratify=y, random_state=42
)
X_set1, X_set2, y_set1, y_set2 = train_test_split(
    X_temp, y_temp, test_size=0.313, stratify=y_temp, random_state=42
)

# Train/test splits for each set
def stratified_train_test(X, y, train_ratio):
    return train_test_split(X, y, train_size=train_ratio, stratify=y, random_state=42)

X1_train, X1_test, y1_train, y1_test = stratified_train_test(X_set1, y_set1, 0.75)
X2_train, X2_test, y2_train, y2_test = stratified_train_test(X_set2, y_set2, 0.78)
X3_train, X3_test, y3_train, y3_test = stratified_train_test(X_set3, y_set3, 0.74)

# Save hospital datasets
joblib.dump((X2_train, y2_train, X2_test, y2_test), "HOSPITALS/hospital1/hospital1_data.pkl")
joblib.dump((X3_train, y3_train, X3_test, y3_test), "HOSPITALS/hospital2/hospital2_data.pkl")

print("Dataset split and saved for hospitals.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X1_train, y1_train)

pred = model.predict(X1_test)
acc = accuracy_score(y1_test, pred)
print("Main Model Accuracy:", acc)

# Save main model to central server
joblib.dump(model, "CENTRAL_SERVER/Models/main_model_v1.pkl")
print("Main model saved at CENTRAL_SERVER/Models/main_model_v1.pkl")