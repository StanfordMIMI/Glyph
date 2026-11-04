"""
Evaluation metrics (AUC, accuracy, etc.).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from typing import Union


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        raise ValueError("y_true must contain at least 2 classes")
    return roc_auc_score(y_true, y_pred)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> float:
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_all_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                        threshold: float = 0.5) -> dict:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        'auc': compute_auc(y_true, y_pred_proba),
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred),
        'recall': compute_recall(y_true, y_pred),
        'f1': compute_f1(y_true, y_pred),
    }
