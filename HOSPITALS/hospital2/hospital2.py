import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

hospital_data_path = "HOSPITALS/hospital2/hospital2_data.pkl"
main_model_path = "CENTRAL_SERVER/Models/main_model_v1.pkl"
hospital_model_path = "HOSPITALS/hospital2/hospital2_v2.pkl"

# Load data
X_train, y_train, X_test, y_test = joblib.load(hospital_data_path)

# Load main model
main_model = joblib.load(main_model_path)

# Test initial accuracy on small sample
initial_samples = 50
y_pred_initial = main_model.predict(X_test[:initial_samples])
init_acc = accuracy_score(y_test[:initial_samples], y_pred_initial)
print(f"Initial Accuracy on {initial_samples} samples: {init_acc:.4f}")

# Retrain if accuracy is below main model
if init_acc < 0.97:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    final_pred = model.predict(X_test)
    final_acc = accuracy_score(y_test, final_pred)
    print(f"Final Accuracy on all test samples: {final_acc:.4f}")
    joblib.dump(model, hospital_model_path)
    print(f"Hospital retrained model saved → {hospital_model_path}")
else:
    print("No retraining needed, accuracy is sufficient.")