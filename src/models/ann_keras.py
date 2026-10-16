from pathlib import Path
import sys
import time

import pandas as pd

import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.utils.model_logger import save_model_results


# Data paths
DATA_DIR = BASE_DIR / "data" / "processed" / "model_data"
MODEL_FILE = DATA_DIR / "ann_keras_model.keras"

# Load data
print("\nLoading Training and Validation Data...")

X_train = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
X_val = pd.read_csv(DATA_DIR / "X_val_scaled.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()

print(f"Training data: {X_train.shape}")
print(f"Validation data: {X_val.shape}")


# Build ANN
print("\nBuilding Keras ANN...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)


# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# Train model
print("\nTraining Keras ANN...")

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

training_time = time.time() - start_time


# Save Trained Model
model.save(MODEL_FILE)
print(f"\nKeras ANN Model Saved : {MODEL_FILE}")


# Predictions
print("\nEvaluating Model...")

y_prob = model.predict(X_val, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)


# Metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_prob)
cm = confusion_matrix(y_val, y_pred)


# Print results
print("\nKeras ANN Results")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:\n", cm)


# Save results
save_model_results(
    model_name="ANN (Keras)",
    task="Classification",
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    roc_auc=roc_auc,
    training_time=training_time
)

print("\nKeras ANN Saved Successfully!")

print(f"\nTraining Time: {training_time:.2f} seconds")
