import joblib
import pprint

print("=== model_columns.pkl ===")
columns = joblib.load("../models/model_columns.pkl")
pprint.pprint(columns)

print("\n=== scaler.pkl ===")
scaler = joblib.load("../models/scaler.pkl")
print(f"Scaler type: {type(scaler)}")
print(f"Mean: {scaler.mean_}")
print(f"Scale/Variance: {scaler.scale_}")

print("\n=== model.pkl ===")
model = joblib.load("model.pkl")
print(f"Model type: {type(model)}")
print(f"Number of estimators: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")
print(f"Number of features seen: {model.n_features_in_}")
