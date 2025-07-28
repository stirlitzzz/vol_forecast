# 5. IMPROVED MODEL TRAINING (drop-in replacement):
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
def train_model_with_validation(X_train, y_train, X_val, y_val, param_grid=None):
    """Train model with proper validation instead of using test set"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import roc_auc_score, classification_report
    
    if param_grid is None:
        param_grid = {
            #'max_depth': [3, 5, 7],
            'max_depth': [3],
            'n_estimators': [50, 100],
            'min_samples_split': [2, 5]
        }
    
    # Use validation set for hyperparameter tuning
    #rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    best_score = 0
    best_model = None
    best_params = None
    
    # Manual grid search using validation set
    for max_depth in param_grid['max_depth']:
        for n_estimators in param_grid['n_estimators']:
            for min_samples_split in param_grid['min_samples_split']:
                
                model = RandomForestClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    min_samples_split=min_samples_split,
                    class_weight='balanced',
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                val_pred_proba = model.predict_proba(X_val)[:, 1]
                val_score = roc_auc_score(y_val, val_pred_proba)
                
                if val_score > best_score:
                    best_score = val_score
                    best_model = model
                    best_params = {
                        'max_depth': max_depth,
                        'n_estimators': n_estimators,
                        'min_samples_split': min_samples_split
                    }
    
    print(f"Best validation AUC: {best_score:.4f}")
    print(f"Best params: {best_params}")
    
    return best_model