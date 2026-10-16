from pathlib import Path
import sys
import time

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

from src.utils.model_logger import save_model_results

from tensorflow.keras.models import Sequential                          # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input         # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping                    # pyright: ignore[reportMissingModuleSource]


# PROJECT PATH
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lstm_data"
    / "customer_sequences.npz"
)

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lstm_data"
    / "lstm_model.keras"
)

SCALER_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "lstm_data"
    / "lstm_scaler.pkl"
)


# RANDOM SEED
np.random.seed(42)
tf.random.set_seed(42)


# LOAD LSTM DATA
print("\nLoading LSTM Data...")

data = np.load(INPUT_FILE)

X = data["X"]
y = data["y"]
customer_ids = data["customer_ids"]

print(f"Sequence Shape : {X.shape}")
print(f"Target Shape   : {y.shape}")
print(f"Customers      : {len(np.unique(customer_ids))}")


# CUSTOMER-LEVEL TRAIN / TEST SPLIT
print("\nCreating Customer-Level Train/Test Split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_val_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=customer_ids
    )
)

X_train_val = X[train_val_idx]
y_train_val = y[train_val_idx]
groups_train_val = customer_ids[train_val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

groups_test = customer_ids[test_idx]


# CUSTOMER-LEVEL TRAIN / VALIDATION SPLIT
print("\nCreating Customer-Level Train/Validation Split...")

splitter_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1875,
    random_state=42
)

train_idx, val_idx = next(
    splitter_val.split(
        X_train_val,
        y_train_val,
        groups=groups_train_val
    )
)

X_train = X_train_val[train_idx]
y_train = y_train_val[train_idx]

X_val = X_train_val[val_idx]
y_val = y_train_val[val_idx]

print(f"\nTraining Sequences : {len(X_train)}")
print(f"Validation Sequences : {len(X_val)}")
print(f"Testing Sequences    : {len(X_test)}")

print(f"Training Customers   : {len(np.unique(groups_train_val[train_idx]))}")
print(f"Validation Customers : {len(np.unique(groups_train_val[val_idx]))}")
print(f"Testing Customers    : {len(np.unique(groups_test))}")


# FEATURE SCALING
print("\nScaling LSTM Features...")

train_shape = X_train.shape
val_shape = X_val.shape
test_shape = X_test.shape


# Convert 3D sequences into 2D
X_train_2d = X_train.reshape(-1, X_train.shape[2])
X_val_2d = X_val.reshape(-1, X_val.shape[2])
X_test_2d = X_test.reshape(-1, X_test.shape[2])


# Fit scaler ONLY on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_2d)
X_val_scaled = scaler.transform(X_val_2d)
X_test_scaled = scaler.transform(X_test_2d)


# Convert back to 3D
X_train = X_train_scaled.reshape(train_shape)
X_val = X_val_scaled.reshape(val_shape)
X_test = X_test_scaled.reshape(test_shape)

print("\nFeature Scaling Completed!")


# BUILD LSTM MODEL
print("\nBuilding LSTM Model...")

model = Sequential([
    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    LSTM(64),

    Dropout(0.20),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1
    )
])


# COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# EARLY STOPPING
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# TRAIN MODEL
print("\nTraining LSTM Model...")

start_time = time.time()

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,
    batch_size=256,

    callbacks=[
        early_stopping
    ],

    verbose=1
)

training_time = time.time() - start_time

model.save(MODEL_FILE)
print(f"\nLSTM Model Saved : {MODEL_FILE}")

import joblib
joblib.dump(scaler, SCALER_FILE)
print(f"LSTM Scaler Saved : {SCALER_FILE}")


# TEST SET PREDICTION
print("\nEvaluating LSTM Model...")

y_pred = model.predict(X_test, verbose=0).flatten()


# REGRESSION METRICS
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# DISPLAY RESULTS
print("\nLSTM Results")
print("-------------------------------------")
print(f"MAE             : {mae:.4f}")
print(f"RMSE            : {rmse:.4f}")
print(f"R2 Score        : {r2:.4f}")
print(f"Training Time   : {training_time:.2f} sec")


# SAVE MODEL RESULTS
save_model_results(
    model_name="LSTM",
    mae=mae,
    rmse=rmse,
    r2=r2,
    training_time=training_time,
    task="Regression"
)

print("\nLSTM Results Saved Successfully!")