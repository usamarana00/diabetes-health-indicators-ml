"""
evaluation.py
Helper functions for plotting model evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)


def plot_roc_curves(models: dict, X_test, y_test, title='ROC Curves'):
    """Plot ROC curves for multiple binary classifiers."""
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, class_names, title='Confusion Matrix'):
    """Plot confusion matrix for a classifier."""
    cm = confusion_matrix(y_test, model.predict(X_test))

    plt.figure(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs Predicted'):
    """Scatter plot of actual vs predicted values for regression."""
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.3, color='steelblue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def regression_metrics(y_true, y_pred, model_name='Model'):
    """Return a dict of regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    return {
        'Model': model_name,
        'MAE':  round(mean_absolute_error(y_true, y_pred), 4),
        'MSE':  round(mse, 4),
        'RMSE': round(np.sqrt(mse), 4),
        'R2':   round(r2_score(y_true, y_pred), 4)
    }
