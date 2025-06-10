import tkinter as tk
from data_utils import load_ac_dataset, preprocess_ac_data
from model_utils import prepare_features, train_and_tune_models
from gui_app import ACAssistantApp

def main():
    # Load and preprocess dataset
    df, load_msg = load_ac_dataset()
    df = preprocess_ac_data(df)

    # Prepare features and target
    X_encoded, y, column_transformer = prepare_features(df)

    # Train models and create ensemble
    (ensemble_model, ensemble_r2, rf_params, rf_r2,
     xgb_params, xgb_r2, X_train, X_test, y_train, y_test) = train_and_tune_models(X_encoded, y)

    # Initialize GUI app
    root = tk.Tk()
    app = ACAssistantApp(root, df, column_transformer, ensemble_model, rf_r2, xgb_r2, ensemble_r2, load_msg)
    root.mainloop()

if __name__ == "__main__":
    main()
