import os
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.stats import randint as sp_randint, loguniform
import sqlite3
from datetime import datetime
import numpy as np

# --- Data Loading and Splitting Utility (Conceptual src/data.py) ---

def load_and_split_data(test_size=0.2, random_state=42):
    DB_PATH = '../../data/db/obesity_data_processed.db'
    
    try:
        conn = sqlite3.connect(DB_PATH)
        processed_data = pd.read_sql_query("SELECT * FROM obesity_data_processed", conn)
        conn.close()
    except sqlite3.Error as e:
        print(f"Error loading data from database: {e}")
        return None, None, None, None

    features = processed_data.drop(["NObeyesdad", "timestamp"], axis=1)
    target = processed_data["NObeyesdad"]
    
    # FIX: Reverse the MinMax scaling on the target variable
    target = (target * 6).round().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )
    print("Data loaded, target corrected, and split successfully.")
    return X_train, X_test, y_train, y_test

# --- Logging Utility (Conceptual src/logger.py) ---

def log_predictions(model_name: str, y_true, y_pred: np.ndarray, source: str = "train"):
    DB_PATH = '../../data/db/split_data.db'
    
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    actual_values = y_true.values if isinstance(y_true, pd.Series) else y_true

    log_df = pd.DataFrame({
        "timestamp": [datetime.now().isoformat()] * len(y_pred),
        "model": [model_name] * len(y_pred),
        "source": [source] * len(y_pred),
        "actual": actual_values,
        "predicted": y_pred
    })
    
    try:
        conn = sqlite3.connect(DB_PATH)
        log_df.to_sql("predictions", conn, if_exists="append", index=False)
        conn.close()
        print(f"   -> Predictions for {model_name} ({source}) logged.")
    except sqlite3.Error as e:
        print(f"   -> Error logging predictions for {model_name} ({source}): {e}")


# --- Main Execution (Conceptual labs/lab3/main.py) ---

# 1. Load Data
X_train, X_test, y_train, y_test = load_and_split_data()
if X_train is None:
    raise RuntimeError("Failed to load and split data.")

# 2. Initial Model Definitions
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs'),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "CatBoostClassifier": CatBoostClassifier(verbose=0, random_state=42, allow_writing_files=False, loss_function='MultiClass', eval_metric='MultiClass'),
    "SVC": SVC(random_state=42, class_weight='balanced', probability=True)
}
    
# 3. Initial Training, Logging, and Evaluation (Baseline)
print("\n--- Initial Training & Evaluation (Baseline) ---")
initial_metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    log_predictions(name, y_test, y_pred_test, source="test")
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    initial_metrics[name] = {"Test Accuracy": test_accuracy, "Test F1-score (Weighted)": test_f1_weighted}

print("\nInitial Test Set Performance:")
print(pd.DataFrame.from_dict(initial_metrics, orient='index').round(4))
print("\n" + "="*50)


# 4. Hyperparameter Optimization Setup
os.makedirs("models", exist_ok=True)
print("✅ 'models/' directory ensured.")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metric = 'f1_weighted'

optimized_models = {}

# 5.1 Random Forest Optimization
print("\n--- Starting Randomized Search for Random Forest ---")
rf_param_dist = {
    'n_estimators': sp_randint(100, 500), 'max_depth': sp_randint(10, 30),
    'min_samples_split': sp_randint(2, 15), 'min_samples_leaf': sp_randint(1, 10),
}
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=rf_param_dist, n_iter=20, cv=cv_strategy, scoring=scoring_metric,
    verbose=0, random_state=42, n_jobs=-1
)
rf_random_search.fit(X_train, y_train)
best_rf_model = rf_random_search.best_estimator_
optimized_models["RandomForestClassifier"] = best_rf_model
print(f"Best RF CV Score: {rf_random_search.best_score_:.4f}")
joblib.dump(best_rf_model, 'models/RandomForest_optimized_model.pkl')

# 5.2 CatBoost Optimization
print("\n--- Starting Randomized Search for CatBoost ---")
cb_param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2], 'depth': sp_randint(4, 10),
    'l2_leaf_reg': [1, 3, 5, 7, 9], 'iterations': [100, 300, 500]
}
cb_random_search = RandomizedSearchCV(
    estimator=CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False, loss_function='MultiClass'),
    param_distributions=cb_param_dist, n_iter=15, cv=cv_strategy, scoring=scoring_metric,
    verbose=0, random_state=42, n_jobs=-1
)
cb_random_search.fit(X_train, y_train)
best_cb_model = cb_random_search.best_estimator_
optimized_models["CatBoostClassifier"] = best_cb_model
print(f"Best CatBoost CV Score: {cb_random_search.best_score_:.4f}")
joblib.dump(best_cb_model, 'models/CatBoost_optimized_model.pkl')

# 5.3 SVC Optimization
print("\n--- Starting Randomized Search for SVC ---")
svc_param_dist = {
    'C': loguniform(1e-1, 1e2), 'gamma': loguniform(1e-3, 1),
    'kernel': ['rbf', 'poly'], 'degree': [2, 3] 
}
svc_random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42, class_weight='balanced'),
    param_distributions=svc_param_dist, n_iter=20, cv=cv_strategy, scoring=scoring_metric,
    verbose=0, random_state=42, n_jobs=-1
)
svc_random_search.fit(X_train, y_train)
best_svc_model = svc_random_search.best_estimator_
optimized_models["SVC"] = best_svc_model
print(f"Best SVC CV Score: {svc_random_search.best_score_:.4f}")
joblib.dump(best_svc_model, 'models/SVC_optimized_model.pkl')


# 6. Final Evaluation of Optimized Models and Saving Overall Best
print("\n--- Final Comparison (Optimized Test Set Performance) ---")

final_comparison = {}
best_f1 = -1
best_model_name = ""
best_model = None

for name, model in optimized_models.items():
    y_pred_test = model.predict(X_test)
    
    log_predictions(f"{name}_Optimized", y_test, y_pred_test, source="test")
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    
    final_comparison[name] = {"Test Accuracy": test_accuracy, "Test F1-score (Weighted)": test_f1_weighted}

    if test_f1_weighted > best_f1:
        best_f1 = test_f1_weighted
        best_model_name = name
        best_model = model

final_df = pd.DataFrame.from_dict(final_comparison, orient='index').round(4)
print(final_df)

# 7. Save Overall Best Model
if best_model:
    joblib.dump(best_model, f'../../models/final_model_{best_model_name.lower()}.pkl')
    print(f"\n✨ The overall best model, saved as 'models/final_model_{best_model_name.lower()}.pkl', is the {best_model_name} Classifier.")
    
print("\nRefactoring, optimization, and saving complete.")