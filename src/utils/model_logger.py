import os
import pandas as pd


# Path
RESULTS_DIR = "data/processed/model_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "model_selection_results.csv")


def save_model_results(
    model_name,
    task="Classification",
    accuracy=None,
    precision=None,
    recall=None,
    f1=None,
    roc_auc=None,
    mae=None,
    rmse=None,
    r2=None,
    training_time=None
):
    """Save model evaluation results.
    If the same model and task already exist, the previous entry is replaced instead of creating a duplicate."""

    os.makedirs(RESULTS_DIR, exist_ok=True)

    new_result = {
        "Model": model_name,
        "Task": task,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Training Time": training_time
    }

    # Load existing results
    if os.path.exists(RESULTS_FILE):
        results = pd.read_csv(RESULTS_FILE)
    else:
        results = pd.DataFrame()

    # Remove existing entry for same Model + Task
    if not results.empty:
        results = results[
            ~(
                (results["Model"] == model_name) &
                (results["Task"] == task)
            )
        ]

    # Add new result
    results = pd.concat(
        [results, pd.DataFrame([new_result])],
        ignore_index=True
    )

    # Save clean results
    results.to_csv(RESULTS_FILE, index=False)

    print(f"Model results saved: {model_name} | {task}")