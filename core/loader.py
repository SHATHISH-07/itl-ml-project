import pandas as pd
import torch
import os
import threading
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Thread safety lock for model swapping
resource_lock = threading.Lock()

# Global application state
df = None
training_dataset = None
val_loader = None
tft_model = None
tokenizer = None
llm_model = None

# Constants
targets = ["Project_Score", "Tasks_Completed", "Hours_Worked", "Overtime_Hours", "Peer_Feedback", "Attendance"]
max_encoder_length = 30
max_prediction_length = 4
min_encoder_length = 15

def load_all_resources():
    global tokenizer, llm_model
    if llm_model is None:
        print("Loading LLM...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="auto")
    reload_data_and_tft()

def reload_data_and_tft():
    global df, training_dataset, val_loader, tft_model
    from services.training_service import preprocess_data, create_datasets # Avoid circular import
    
    data_path = "data/employee_data.json"
    if not os.path.exists(data_path): return

    new_df = pd.read_json(data_path)
    new_df = preprocess_data(new_df)
    new_train_ds, _, _, new_val_loader = create_datasets(new_df)

    model_path = "model/tft_model.ckpt"
    new_tft = None
    if os.path.exists(model_path):
        new_tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
        new_tft.eval()

    # Critical Section: Swap references safely
    with resource_lock:
        df = new_df
        training_dataset = new_train_ds
        val_loader = new_val_loader
        tft_model = new_tft
    print("Resources updated and locked safely.")