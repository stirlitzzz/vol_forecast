import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def split_train_test_multiple_with_validation(*dfs, train_ratio=0.6, val_ratio=0.2, train_test_gap_days=21):
    """Add validation set to your existing split function"""
    # Keep your existing logic but add validation set
    unique_dates = dfs[0].index.get_level_values(0).unique().sort_values()
    
    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end + train_test_gap_days:val_end]
    test_dates = unique_dates[val_end + train_test_gap_days:]
    
    print(f"Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Val: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")  
    print(f"Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for df in dfs:
        if (df is not None):
            train_dfs.append(df[df.index.get_level_values(0).isin(train_dates)])
            val_dfs.append(df[df.index.get_level_values(0).isin(val_dates)])
            test_dfs.append(df[df.index.get_level_values(0).isin(test_dates)])
        else:
            train_dfs.append(None)
            val_dfs.append(None)
            test_dfs.append(None)
    
    return train_dfs, val_dfs, test_dfs