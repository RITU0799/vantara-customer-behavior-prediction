from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model                       # type: ignore
from tensorflow.keras.layers import Input, Dense                # type: ignore
from tensorflow.keras.callbacks import EarlyStopping            # type: ignore


# PROJECT PATH
BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(BASE_DIR))


# IMPORT MODEL LOGGER
from src.utils.model_logger import save_model_results


# FILE PATHS
INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_ready_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "autoencoder"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "customer_anomalies.csv"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "autoencoder.keras"
)

SCALER_FILE = (
    OUTPUT_DIR
    / "autoencoder_scaler.pkl"
)

THRESHOLD_FILE = (
    OUTPUT_DIR
    / "anomaly_threshold.npy"
)


# RANDOM SEED
np.random.seed(42)
tf.random.set_seed(42)


# LOAD DATA
print("\nLoading Autoencoder Data...")

df = pd.read_csv(INPUT_FILE)
print(f"Dataset Shape : {df.shape}")


# SELECT CUSTOMER ID
customer_ids = df["Customer_ID"].copy()


# SELECT NUMERIC FEATURES
exclude_columns = ["Customer_ID", "Churn"]

feature_columns = [
    column
    for column in df.select_dtypes(include=np.number).columns
    if column not in exclude_columns
]

X = df[feature_columns].copy()

print(f"Features Used : {len(feature_columns)}")
print(f"Feature Shape : {X.shape}")


# HANDLE MISSING / INFINITE VALUES
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# TRAIN / TEST SPLIT
print("\nCreating Train/Test Split...")

X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")


# FEATURE SCALING
print("\nScaling Features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed!")


# BUILD AUTOENCODER
print("\nBuilding Autoencoder Model...")

input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))


# Encoder
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)

# Decoder
decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)


# COMPILE MODEL
autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


# EARLY STOPPING
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# TRAIN AUTOENCODER
print("\nTraining Autoencoder Model...")

start_time = time.time()

history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled,
    validation_split=0.20,
    epochs=50,
    batch_size=256,
    callbacks=[
        early_stopping
    ],
    verbose=1
)

training_time = time.time() - start_time


# RECONSTRUCT TEST DATA
print("\nCalculating Reconstruction Error...")

X_test_reconstructed = autoencoder.predict(X_test_scaled, verbose=0)


# CALCULATE RECONSTRUCTION ERROR
reconstruction_error = np.mean(np.square(X_test_scaled - X_test_reconstructed), axis=1)


# ANOMALY THRESHOLD
threshold = np.percentile(reconstruction_error, 95)

autoencoder.save(MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
np.save(THRESHOLD_FILE, threshold)

print(f"\nAutoencoder Model Saved : {MODEL_FILE}")
print(f"Autoencoder Scaler Saved : {SCALER_FILE}")
print(f"Anomaly Threshold Saved : {THRESHOLD_FILE}")

# ANOMALY FLAG
anomaly_flag = (reconstruction_error > threshold).astype(int)


# SAVE CUSTOMER ANOMALY RESULTS
results = pd.DataFrame({
    "Customer_ID": customer_ids.loc[X_test.index].values,
    "Reconstruction_Error": reconstruction_error,
    "Anomaly": anomaly_flag
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT_FILE, index=False)


# CALCULATE SUMMARY
anomaly_count = anomaly_flag.sum()

anomaly_percentage = (anomaly_count / len(anomaly_flag)) * 100


# DISPLAY RESULTS
print("\nAutoencoder Results")
print("-------------------------------------")
print(f"Features Used        : {len(feature_columns)}")
print(f"Test Customers       : {len(X_test)}")
print(f"Anomalies Detected   : {anomaly_count}")
print(f"Anomaly Percentage   : {anomaly_percentage:.2f}%")
print(f"Threshold            : {threshold:.6f}")
print(f"Training Time        : {training_time:.2f} sec")


# SAVE MODEL RESULT TO LOGGER
save_model_results(
    model_name="Autoencoder",
    training_time=training_time,
    task="Anomaly Detection"
)


# FINAL MESSAGE
print("\nAutoencoder Results Saved Successfully!")

print(f"Customer Results : {OUTPUT_FILE}")

print("\nModel Log : data/processed/model_results/model_selection_results.csv")

