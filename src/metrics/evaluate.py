# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.preprocess import feature_engineering
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


def plot_feature_importance(pipeline, feature_names, top_n=20):
    model = pipeline.named_steps['model']

    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = model.coef_[0]
        importances = np.abs(importances)

    indices = np.argsort(importances)[-top_n:]
    engineered_feature_names = feature_engineering(feature_names.copy()).columns

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [engineered_feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top Important Features')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()





def results(pipeline, X_test, y_test, X_train):
    """
    Print classification report, confusion matrix, ROC curve,
    and feature importance plot for a trained pipeline.
    """
    # --- Classification report ---
    y_pred = pipeline.predict(X_test)
    print('Classification Report:\n', classification_report(y_true=y_test, y_pred=y_pred))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # --- ROC curve ---
    if hasattr(pipeline.named_steps['model'], "predict_proba"):
        y_preds = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

    # --- Feature importance ---
    plot_feature_importance(pipeline, feature_names=X_train)

