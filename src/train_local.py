import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Check if dataset exists
try:
    df = pd.read_csv('crop_yield.csv')
except FileNotFoundError:
    print("Please download 'crop_yield.csv' into this folder first from Kaggle!")
    exit()

print("Loaded Data. Processing...")

# Preprocessing to match Colab exactly
df.drop('Region', axis=1, inplace=True, errors='ignore')
df.drop('Weather_Condition', axis=1, inplace=True, errors='ignore')

# Use 100k rows to speed up local retraining (Colab used 1M)
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42).reset_index(drop=True)

df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)

features_to_scale = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

df = pd.get_dummies(df, columns=['Soil_Type', 'Crop'])
df = df.replace({True: 1, False: 0})

X = df.drop('Yield_tons_per_hectare', axis=1)
y = df['Yield_tons_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print(f"R2 Score: {rf_model.score(X_test, y_test):.4f}")

print("Saving Models...")
joblib.dump(rf_model, '../models/model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
joblib.dump(X_train.columns.tolist(), '../models/model_columns.pkl')

print("\n🚀 Models saved successfully! You can now start the Streamlit app.")
