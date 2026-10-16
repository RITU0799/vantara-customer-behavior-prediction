import pandas as pd
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_results"
    / "final_clv_test_results.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_results"
    / "clv_final_comparison.csv"
)


def main():
    """Create the final CLV model comparison."""

    print("\nLoading CLV Test Results...")
    print("-------------------------------------")

    results = pd.read_csv(RESULTS_PATH)
    print(f"Input Shape : {results.shape}")

    print("\nAvailable Columns")
    print("-------------------------------------")
    print(results.columns.tolist())

    comparison = results[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    ].copy()

    # PRD target
    comparison["PRD_Target_R2"] = 0.60

    comparison["Target_Met"] = (
        comparison["R2"] >= comparison["PRD_Target_R2"]
    )

    # Sort by R²
    comparison = comparison.sort_values(
        by="R2",
        ascending=False
    ).reset_index(drop=True)

    # Save final comparison
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFinal CLV Model Comparison")
    print("-------------------------------------")
    print(comparison.to_string(index=False))

    print("\nSelected CLV Model")
    print("-------------------------------------")

    best_model = comparison.iloc[0]

    print(f"Model : {best_model['Model']}")
    print(f"MAE   : {best_model['MAE']:.4f}")
    print(f"RMSE  : {best_model['RMSE']:.4f}")
    print(f"R²    : {best_model['R2']:.4f}")

    if best_model["R2"] >= 0.60:
        print("PRD R² Target : MET")
    else:
        print("PRD R² Target : NOT MET")

    print(
        f"\nFinal comparison saved to : "
        f"{OUTPUT_PATH}"
    )

    print("\nCLV Final Comparison Completed Successfully!")

if __name__ == "__main__":
    main()