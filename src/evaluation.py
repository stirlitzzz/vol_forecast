import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_model_diagnostics(y_true, y_pred_proba, title="Model Diagnostics"):
    """Quick diagnostic plots for your model"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[1].plot(recall, precision)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    
    # Prediction Distribution by Class
    axes[2].hist(y_pred_proba[y_true == 0], alpha=0.5, label='Class 0', bins=30)
    axes[2].hist(y_pred_proba[y_true == 1], alpha=0.5, label='Class 1', bins=30)
    axes[2].set_xlabel('Predicted Probability')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Prediction Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    return fig