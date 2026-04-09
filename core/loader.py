import pandas as pd
import torch
import os
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from services.training_service import preprocess_data, create_datasets

# Global application state
df = None
training_dataset = None
val_loader = None
tft_model = None
tokenizer = None
llm_model = None

# Constants
targets = [
    "Project_Score", "Tasks_Completed", "Hours_Worked",
    "Overtime_Hours", "Peer_Feedback", "Attendance"
]
max_encoder_length = 30
max_prediction_length = 4
min_encoder_length = 15

def load_all_resources():
    """Initial load of data, models, and LLM on server startup."""
    global tokenizer, llm_model
    
    # Only load the LLM once to save time
    if llm_model is None:
        print("Loading Qwen LLM... (This will take a few minutes to download the first time!)")
        
        # Switched to 0.5B to make your local download much faster
        model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Using torch_dtype="auto" is the standard, non-deprecated way for Hugging Face
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.float32, 
            device_map="auto"
        )
    
    reload_data_and_tft()

def reload_data_and_tft():
    global df, training_dataset, val_loader, tft_model
    
    data_path = "data/employee_data.json"
    if not os.path.exists(data_path):
        print("No data found. Upload data via /api/model/train endpoint.")
        return

    print("Loading and Preprocessing Data...")
    df = pd.read_json(data_path)
    df = preprocess_data(df)
    
    print("Rebuilding Datasets...")
    training_dataset, validation_dataset, _, v_loader = create_datasets(df)
    val_loader = v_loader

    print("Loading TFT Model Checkpoint...")
    model_path = "model/tft_model.ckpt"
    if os.path.exists(model_path):
        tft_model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        tft_model.eval()
        print("TFT Model loaded successfully!")
    else:
        print("No trained TFT model found. Please run training.")