"""Hyperparameter optimization utilities."""

import os
import joblib
from typing import Dict, Any, Optional

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from scipy.stats import randint as sp_randint, loguniform


def setup_cv_strategy(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Create cross-validation strategy."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )


def optimize_random_forest(
    X_train: Any,
    y_train: Any,
    cv_strategy: StratifiedKFold,
    n_iter: int = 20,
    random_state: int = 42,
    save_path: Optional[str] = 'models/RandomForest_optimized_model.pkl'
) -> RandomForestClassifier:
    """Optimize RandomForest hyperparameters using RandomizedSearchCV."""
    param_dist = {
        'n_estimators': sp_randint(100, 500),
        'max_depth': sp_randint(10, 30),
        'min_samples_split': sp_randint(2, 15),
        'min_samples_leaf': sp_randint(1, 10),
    }
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_state, class_weight='balanced'),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring='f1_weighted',
        verbose=0,
        random_state=random_state,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best RF CV Score: {random_search.best_score_:.4f}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(random_search.best_estimator_, save_path)
    
    return random_search.best_estimator_


def optimize_catboost(
    X_train: Any,
    y_train: Any,
    cv_strategy: StratifiedKFold,
    n_iter: int = 15,
    random_state: int = 42,
    save_path: Optional[str] = 'models/CatBoost_optimized_model.pkl'
) -> CatBoostClassifier:
    """Optimize CatBoost hyperparameters using RandomizedSearchCV."""
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'depth': sp_randint(4, 10),
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'iterations': [100, 300, 500]
    }
    
    random_search = RandomizedSearchCV(
        estimator=CatBoostClassifier(
            random_state=random_state,
            verbose=0,
            allow_writing_files=False,
            loss_function='MultiClass'
        ),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring='f1_weighted',
        verbose=0,
        random_state=random_state,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best CatBoost CV Score: {random_search.best_score_:.4f}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(random_search.best_estimator_, save_path)
    
    return random_search.best_estimator_


def optimize_svc(
    X_train: Any,
    y_train: Any,
    cv_strategy: StratifiedKFold,
    n_iter: int = 20,
    random_state: int = 42,
    save_path: Optional[str] = 'models/SVC_optimized_model.pkl'
) -> SVC:
    """Optimize SVC hyperparameters using RandomizedSearchCV."""
    param_dist = {
        'C': loguniform(1e-1, 1e2),
        'gamma': loguniform(1e-3, 1),
        'kernel': ['rbf', 'poly'],
        'degree': [2, 3]
    }
    
    random_search = RandomizedSearchCV(
        estimator=SVC(random_state=random_state, class_weight='balanced'),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring='f1_weighted',
        verbose=0,
        random_state=random_state,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best SVC CV Score: {random_search.best_score_:.4f}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(random_search.best_estimator_, save_path)
    
    return random_search.best_estimator_


def save_best_model(
    model: Any,
    model_name: str,
    save_dir: str = '../../models'
) -> None:
    """Save the best performing model."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'final_model_{model_name.lower()}.pkl')
    joblib.dump(model, save_path)
    print(f"\n✨ The overall best model, saved as '{save_path}', is the {model_name} Classifier.")