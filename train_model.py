import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# 1. LOAD DATA
try:
    df = pd.read_csv('data/processed_train.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('processed_train.csv')
    except:
        df = pd.read_csv('AeroGuard/data/processed_train.csv')

print(f"Data Loaded. Shape: {df.shape}")

# 2. FEATURE ENGINEERING (The "Secret Sauce")
sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
           's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# A. Rolling Mean & Std Dev (Window = 6 captures more history)
window_size = 6
print(f"Generating Rolling Features (Window={window_size})...")

rolling_mean = df.groupby('unit_nr')[sensors].rolling(window=window_size).mean().reset_index(level=0, drop=True)
rolling_mean.columns = [f"{s}_mean" for s in sensors]

rolling_std = df.groupby('unit_nr')[sensors].rolling(window=window_size).std().reset_index(level=0, drop=True)
rolling_std.columns = [f"{s}_std" for s in sensors]

# B. LAG FEATURES (New! - Captures immediate past values)
# "What was the value 1 cycle ago? 2 cycles ago?"
print("Generating Lag Features...")
lag_df = df.copy()
for s in ['s_11', 's_12', 's_4', 's_7']: # Only lag the most critical sensors
    lag_df[f'{s}_lag1'] = df.groupby('unit_nr')[s].shift(1)
    lag_df[f'{s}_lag2'] = df.groupby('unit_nr')[s].shift(2)

# Select only the lag columns
lag_cols = [c for c in lag_df.columns if '_lag' in c]
lag_features = lag_df[lag_cols]

# Combine everything
df_final = pd.concat([df, rolling_mean, rolling_std, lag_features], axis=1)
df_final = df_final.dropna() # Drop rows with NaNs (first few cycles)

# 3. TRAIN/TEST SPLIT
features = list(rolling_mean.columns) + list(rolling_std.columns) + lag_cols
X = df_final[features]
y = df_final['RUL']

# ⚠️ CRITICAL: Clip RUL for training
# NASA dataset RUL is linear, but real engines don't "degrade" in the first 100 cycles.
# Capping RUL at 125 helps the model focus on the *end* of life, where it matters.
y = y.clip(upper=125) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. XGBOOST TRAINING
print("🚀 Training XGBoost Regressor...")
# These parameters are tuned for the CMAPSS dataset
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.5,
    colsample_bytree=0.5,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# 5. EVALUATE
preds = xgb_model.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\n🎯 XGBoost Accuracy (R^2): {r2:.2%}") # Should be closer to 85%
print(f"📉 Root Mean Squared Error: {rmse:.2f} cycles")

# 6. SAVE
joblib.dump(xgb_model, 'aero_guard_model.pkl')
print("💾 XGBoost Model saved as 'aero_guard_model.pkl'")