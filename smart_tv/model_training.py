import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

def train_models(TV_X, TV_y):
    TV_X_train, TV_X_test, TV_y_train, TV_y_test = train_test_split(TV_X, TV_y, test_size=0.2, random_state=42)

    # RandomForest hyperparameter tuning
    TV_rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    TV_rf_random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42), TV_rf_param_grid,
        n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42
    )
    TV_rf_random_search.fit(TV_X_train, TV_y_train)
    TV_rf_best_params = TV_rf_random_search.best_params_
    TV_rf_mean_r2 = np.mean(cross_val_score(TV_rf_random_search.best_estimator_, TV_X_train, TV_y_train, cv=5, scoring='r2'))

    # XGBoost hyperparameter tuning
    TV_xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_child_weight': [1, 3, 5]
    }
    TV_xgb_random_search = RandomizedSearchCV(
        XGBRegressor(random_state=42), TV_xgb_param_grid,
        n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42
    )
    TV_xgb_random_search.fit(TV_X_train, TV_y_train)
    TV_xgb_best_params = TV_xgb_random_search.best_params_
    TV_xgb_mean_r2 = np.mean(cross_val_score(TV_xgb_random_search.best_estimator_, TV_X_train, TV_y_train, cv=5, scoring='r2'))

    # Ensemble model
    TV_ensemble_model = VotingRegressor([
        ('RandomForest', TV_rf_random_search.best_estimator_),
        ('XGBoost', TV_xgb_random_search.best_estimator_)
    ])
    TV_ensemble_model.fit(TV_X_train, TV_y_train)

    return {
        'rf_params': TV_rf_best_params,
        'rf_score': TV_rf_mean_r2,
        'xgb_params': TV_xgb_best_params,
        'xgb_score': TV_xgb_mean_r2,
        'ensemble_model': TV_ensemble_model
    }
