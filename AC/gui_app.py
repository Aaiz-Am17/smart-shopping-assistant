import tkinter as tk
import pandas as pd
from data_utils import preprocess_ac_data
from model_utils import prepare_features

class ACAssistantApp:
    def __init__(self, root, df, column_transformer, ensemble_model, rf_r2, xgb_r2, ensemble_r2, load_message):
        self.root = root
        self.df = df
        self.column_transformer = column_transformer
        self.ensemble_model = ensemble_model
        self.rf_r2 = rf_r2
        self.xgb_r2 = xgb_r2
        self.ensemble_r2 = ensemble_r2
        self.load_message = load_message
        self.root.title("Smart Air Conditioner Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg="#f5f5f5")
        self.create_main_screen()

    def create_main_screen(self):
        frame = tk.Frame(self.root, bg="#f5f5f5")
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Smart Air Conditioner Assistant", font=("Arial", 20, "bold"), bg="#f5f5f5", fg="#34495e").pack(pady=20)
        tk.Label(frame, text=self.load_message, font=("Arial", 12), bg="#f5f5f5", fg="#2c3e50").pack(pady=10)
        tk.Label(frame, text=f"Best RandomForest R2: {self.rf_r2:.3f}", font=("Arial", 12), bg="#f5f5f5").pack(pady=5)
        tk.Label(frame, text=f"Best XGBoost R2: {self.xgb_r2:.3f}", font=("Arial", 12), bg="#f5f5f5").pack(pady=5)
        tk.Label(frame, text=f"Ensemble Model R2: {self.ensemble_r2:.3f}", font=("Arial", 12), bg="#f5f5f5").pack(pady=5)
        start_button = tk.Button(frame, text="Start Prediction", command=self.remove_loading_screen)
        start_button.pack(pady=20)

    def remove_loading_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_category_screen()

    def create_category_screen(self):
        frame = tk.Frame(self.root, bg="#f5f5f5")
        frame.pack(fill="both", expand=True)

        categories = {
            "Condenser_Coil": self.df['Condenser_Coil'].unique(),
            "Refrigerant": self.df['Refrigerant'].unique(),
            "Power_Consumption": ['Low', 'Medium', 'High'],
            "Noise_level": ['Low', 'Medium', 'High']
        }

        self.selected_entries = {}

        for category, options in categories.items():
            tk.Label(frame, text=f"Select {category}:", font=("Arial", 14), bg="#f5f5f5").pack(pady=10)
            button_frame = tk.Frame(frame, bg="#f5f5f5")
            button_frame.pack(fill="x", pady=5)

            for option in options:
                btn = tk.Button(button_frame, text=option,
                                command=lambda c=category, o=option: self.on_option_selected(c, o))
                btn.pack(side="left", padx=5)

    def on_option_selected(self, category, option):
        self.selected_entries[category] = option
        if len(self.selected_entries) == 4:
            self.finalize_input()

    def finalize_input(self):
        input_data = {k: v for k, v in self.selected_entries.items()}
        input_df = pd.DataFrame([input_data])
        input_df = preprocess_ac_data(input_df, for_prediction=True)
        input_encoded = self.column_transformer.transform(input_df)
        predicted_price = self.ensemble_model.predict(input_encoded)

        for widget in self.root.winfo_children():
            widget.destroy()

        result_frame = tk.Frame(self.root, bg="#f5f5f5")
        result_frame.pack(fill="both", expand=True)
        tk.Label(result_frame, text=f"Predicted Price: {predicted_price[0]:.2f}", font=("Arial", 14), bg="#f5f5f5").pack(pady=20)
        restart_btn = tk.Button(result_frame, text="Restart", command=self.restart)
        restart_btn.pack(pady=20)

    def restart(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_main_screen()
