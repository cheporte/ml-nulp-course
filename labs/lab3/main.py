"""Lab 3: Main script for model training and optimization.

This script orchestrates the model training, optimization, and evaluation
process using the modularized components from the lab3 package.
"""

from typing import Dict, Any
import pandas as pd

from data import load_and_split_data
from logger import log_predictions
from training import get_base_models, evaluate_model
from optimization import (
    setup_cv_strategy,
    optimize_random_forest,
    optimize_catboost,
    optimize_svc,
    save_best_model
)


def main():
    # 1. Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()
    if X_train is None:
        raise RuntimeError("Failed to load and split data.")

    # 2. Train and evaluate baseline models
    print("\n--- Initial Training & Evaluation (Baseline) ---")
    models = get_base_models()
    initial_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        log_predictions(name, y_test, y_pred_test, source="test")
        initial_metrics[name] = evaluate_model(y_test, y_pred_test)

    print("\nInitial Test Set Performance:")
    print(pd.DataFrame.from_dict(initial_metrics, orient='index').round(4))
    print("\n" + "="*50)

    # 3. Optimize models
    cv_strategy = setup_cv_strategy()
    optimized_models: Dict[str, Any] = {}

    # 3.1 Random Forest
    print("\n--- Starting Randomized Search for Random Forest ---")
    rf_model = optimize_random_forest(X_train, y_train, cv_strategy)
    optimized_models["RandomForestClassifier"] = rf_model

    # 3.2 CatBoost
    print("\n--- Starting Randomized Search for CatBoost ---")
    cb_model = optimize_catboost(X_train, y_train, cv_strategy)
    optimized_models["CatBoostClassifier"] = cb_model

    # 3.3 SVC
    print("\n--- Starting Randomized Search for SVC ---")
    svc_model = optimize_svc(X_train, y_train, cv_strategy)
    optimized_models["SVC"] = svc_model

    # 4. Final evaluation and model selection
    print("\n--- Final Comparison (Optimized Test Set Performance) ---")
    final_comparison = {}
    best_f1 = -1
    best_model_name = ""
    best_model = None

    for name, model in optimized_models.items():
        y_pred_test = model.predict(X_test)
        log_predictions(f"{name}_Optimized", y_test, y_pred_test, source="test")
        metrics = evaluate_model(y_test, y_pred_test)
        final_comparison[name] = metrics
        
        if metrics["Test F1-score (Weighted)"] > best_f1:
            best_f1 = metrics["Test F1-score (Weighted)"]
            best_model_name = name
            best_model = model

    final_df = pd.DataFrame.from_dict(final_comparison, orient='index').round(4)
    print(final_df)

    # 5. Save best model
    if best_model:
        save_best_model(best_model, best_model_name)
    
    print("\nRefactoring, optimization, and saving complete.")


if __name__ == "__main__":
    main()