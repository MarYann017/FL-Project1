import joblib
import os

os.makedirs("HOSPITALS/hospital1", exist_ok=True)
os.makedirs("HOSPITALS/hospital2", exist_ok=True)

main_model = joblib.load("CENTRAL_SERVER/Models/main_model_v1.pkl")
joblib.dump(main_model, "HOSPITALS/hospital1/hospital1_v2.pkl")
print("Saved hospital1_v2.pkl from main model (no retraining done).")
joblib.dump(main_model, "HOSPITALS/hospital2/hospital2_v2.pkl")
print("Saved hospital2_v2.pkl from main model (no retraining done).")