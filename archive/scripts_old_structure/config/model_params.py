"""
Standardized model hyperparameters.

All scripts must use these parameters to ensure consistency across datasets.
These were determined through extensive tuning and validation.
"""

# Random state for reproducibility
RANDOM_STATE = 42

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Logistic Regression parameters
# C=0.1 for strong regularization (especially for Taiwan with 95 features)
LOGISTIC_PARAMS = {
    'penalty': 'l2',
    'C': 0.1,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'  # Handle class imbalance
}

# Random Forest parameters
# Tuned for bankruptcy prediction (high n_estimators for stability)
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}

# XGBoost parameters
# Conservative learning rate, high iterations for stability
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'scale_pos_weight': None,  # Will be set based on class imbalance
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# LightGBM parameters
# Similar to XGBoost but optimized for LightGBM
LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'verbose': -1
}

# CatBoost parameters
# Similar structure for consistency
CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.01,
    'random_state': RANDOM_STATE,
    'auto_class_weights': 'Balanced',
    'verbose': False
}

# Bootstrap parameters for confidence intervals
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CONFIDENCE = 0.95

# EPV (Events Per Variable) threshold
MIN_EPV = 10.0  # Minimum acceptable EPV for valid inference

# VIF threshold for multicollinearity
MAX_VIF = 10.0  # Variables with VIF > 10 indicate problematic multicollinearity
