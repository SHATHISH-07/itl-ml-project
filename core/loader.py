import pandas as pd
import torch
import os
import threading
from pytorch_forecasting import TemporalFusionTransformer

resource_lock = threading.Lock()

df = None
training_dataset = None
val_loader = None
tft_model = None

targets = [
    "Project_Score",
    "Tasks_Completed",
    "Hours_Worked",
    "Overtime_Hours",
    "Peer_Feedback",
    "Attendance"
]

max_encoder_length = 12
max_prediction_length = 4
min_encoder_length = 2


def load_all_resources():
    load_model_and_data()


def load_model_and_data():
    global df, training_dataset, val_loader, tft_model

    from services.training_service import preprocess_data, create_datasets

    data_path = "data/employee_data.json"
    model_path = "model/best_tft.ckpt"

    if not os.path.exists(data_path):
        print("⚠️ Dataset not found")
        return

    df_new = pd.read_json(data_path)
    df_new = preprocess_data(df_new)

    train_ds, _, _, val_loader_new = create_datasets(df_new)

    model = None
    if os.path.exists(model_path):
        print("✅ Loading trained model...")
        model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        model.eval()
    else:
        print("⚠️ Model not found. Train first.")

    with resource_lock:
        df = df_new
        training_dataset = train_ds
        val_loader = val_loader_new
        tft_model = model

    print(f"✅ Data Loaded: {df is not None}")
    print(f"✅ Model Loaded: {tft_model is not None}")