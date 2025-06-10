import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

def prepare_features(df):
    """
    Extract features and target from preprocessed dataframe.
    Returns:
        X_encoded (numpy array): Encoded feature matrix
        y (Series): Target variable (Price)
        column_transformer (ColumnTransformer): fitted transformer for encoding
    """
    X = df[['Condenser_Coil', 'Refrigerant', 'Power_Consumption', 'Noise_level']]
    y = df['Price']

    categorical_features = ['Condenser_Coil', 'Refrigerant']
    numerical_features = ['Power_Consumption', 'Noise_level']

    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('scaler', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )

    X_encoded = column_transformer.fit_transform(X)
    return X_encoded, y, column_transformer

def train_and_tune_models(X, y):
    """
    Perform train-test split, hyperparameter tuning using RandomizedSearchCV,
    and create an ensemble VotingRegressor.
    Returns:
        ensemble_model: trained VotingRegressor
        test_r2: R^2 score on the test set
        rf_best_params: best params from RandomForest
        rf_cv_r2: cross-validated R^2 for RandomForest
        xgb_best_params: best params from XGBoost
        xgb_cv_r2: cross-validated R^2 for XGBoost
        X_train, X_test, y_train, y_test: split data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, 50],
        'min_samples_split': [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid,
                                   n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    rf_search.fit(X_train, y_train)
    rf_best_params = rf_search.best_params_
    rf_cv_r2 = np.mean(cross_val_score(rf_search.best_estimator_, X_train, y_train, cv=5, scoring='r2'))

    xgb_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid,
                                    n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    xgb_search.fit(X_train, y_train)
    xgb_best_params = xgb_search.best_params_
    xgb_cv_r2 = np.mean(cross_val_score(xgb_search.best_estimator_, X_train, y_train, cv=5, scoring='r2'))

    ensemble_model = VotingRegressor([
        ('RandomForest', rf_search.best_estimator_),
        ('XGBoost', xgb_search.best_estimator_)
    ])
    ensemble_model.fit(X_train, y_train)
    test_r2 = r2_score(y_test, ensemble_model.predict(X_test))

    return (ensemble_model, test_r2, rf_best_params, rf_cv_r2,
            xgb_best_params, xgb_cv_r2, X_train, X_test, y_train, y_test)
