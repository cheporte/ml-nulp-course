"""Model training and evaluation utilities."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from typing import Dict, Any, Union
import numpy as np


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """Get dictionary of base model instances with default configurations.
    
    Args:
        random_state: Random seed for reproducibility.
    
    Returns:
        Dictionary mapping model names to their instances.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            multi_class='multinomial',
            solver='lbfgs'
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
        ),
        "CatBoostClassifier": CatBoostClassifier(
            verbose=0,
            random_state=random_state,
            allow_writing_files=False,
            loss_function='MultiClass',
            eval_metric='MultiClass'
        ),
        "SVC": SVC(
            random_state=random_state,
            class_weight='balanced',
            probability=True
        )
    }


def evaluate_model(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Evaluate model predictions using multiple metrics.
    
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
    
    Returns:
        Dictionary with metric names and their values.
    """
    return {
        "Test Accuracy": accuracy_score(y_true, y_pred),
        "Test F1-score (Weighted)": f1_score(
            y_true, y_pred,
            average='weighted',
            zero_division=0
        )
    }