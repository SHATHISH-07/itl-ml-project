import pandas as pd
import numpy as np
import torch
import datetime
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from core import loader

# Global training status tracker
training_status = {"status": "idle", "message": "Ready"}

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data and generates features required for the TFT model.
    Includes time-based features and performance lags.
    """
    df["Week"] = pd.to_datetime(df["Week"])
    df = df.sort_values(["Employee_ID", "Week"])
    
    # Fundamental Indexing
    df["time_idx"] = df.groupby("Employee_ID").cumcount()
    df["Employee_ID_encoded"] = df["Employee_ID"].astype("category").cat.codes
    
    # Time-varying Known Reals
    df["month"] = df["Week"].dt.month
    df["weekofyear"] = df["Week"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Week"].dt.dayofweek
    df["task_pressure"] = df["Tasks_Completed"] / (df["Hours_Worked"] + 1)
    
    # Lag and Rolling Feature Generation
    # Note: Ensure these match the strings in your time_varying_known_reals list
    for t in loader.targets:
        df[f"lag_1_{t.lower().split('_')[0]}"] = df.groupby("Employee_ID")[t].shift(1)
        df[f"lag_2_{t.lower().split('_')[0]}"] = df.groupby("Employee_ID")[t].shift(2)
        df[f"rolling_{t.lower().split('_')[0]}"] = df.groupby("Employee_ID")[t].transform(lambda x: x.rolling(3).mean())
    
    # Fill gaps created by lagging
    df = df.bfill().ffill()
    return df

def create_datasets(df: pd.DataFrame):
    """
    Prepares training and validation datasets using the 
    advanced MultiNormalizer and Quantile settings.
    """
    # Define the point where training ends and prediction testing begins
    training_cutoff = df["time_idx"].max() - loader.max_prediction_length

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=loader.targets,
        group_ids=["Employee_ID_encoded"],
        max_encoder_length=loader.max_encoder_length,
        min_encoder_length=loader.min_encoder_length,
        max_prediction_length=loader.max_prediction_length,
        static_categoricals=["Department", "Role"],
        time_varying_unknown_reals=loader.targets,
        time_varying_known_reals=[
            "time_idx", "month", "weekofyear", "day_of_week", "task_pressure"
            # Add specific lag/rolling columns here if explicitly needed in model inputs
        ],
        target_normalizer=MultiNormalizer(
            [GroupNormalizer(groups=["Employee_ID_encoded"]) for _ in loader.targets]
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create validation set (predict=True ensures we test the 4-week horizon)
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    return (
        training,
        validation,
        training.to_dataloader(train=True, batch_size=32),
        validation.to_dataloader(train=False, batch_size=32)
    )

def run_training_pipeline():
    """
    Executes the full training loop with EarlyStopping and QuantileLoss.
    """
    global training_status
    try:
        print("🚀 TRAINING STARTED")
        training_status["status"] = "training"

        # 1. Load Data
        df = pd.read_json("data/employee_data.json")
        print(f"📊 Data Loaded: {len(df)} rows")

        # 2. Preprocess and Build Datasets
        df = preprocess_data(df)
        train_ds, val_ds, train_loader, val_loader = create_datasets(df)

        print("⚙️ Model building (Quantile Mode)...")
        
        # 3. Initialize TFT with QuantileLoss
        # Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss_function = QuantileLoss() 
        n_targets = len(loader.targets)
        n_quantiles = len(loss_function.quantiles)

        model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=0.005,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.2,
            hidden_continuous_size=16,
            loss=loss_function,
            # Critical: We must output all quantiles for every target
            output_size=[n_quantiles] * n_targets 
        )

        # 4. Define Trainer with Callbacks
        checkpoint = ModelCheckpoint(
            dirpath="model", 
            filename="best_tft", 
            monitor="val_loss", 
            mode="min"
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=5)

        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="auto",
            callbacks=[checkpoint, early_stop],
            gradient_clip_val=0.1,
            enable_progress_bar=True
        )

        # 5. Run Fit
        print("🏋️ Training running...")
        trainer.fit(model, train_loader, val_loader)

        # 6. Finalize and Save state to loader
        print("✅ Training completed. Loading best weights...")
        best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
        best_model.eval()

        with loader.resource_lock:
            loader.tft_model = best_model
            loader.val_loader = val_loader
            loader.training_dataset = train_ds
            loader.df = df

        training_status["status"] = "completed"
        training_status["message"] = f"Success! Last trained: {datetime.datetime.now().strftime('%H:%M:%S')}"

    except Exception as e:
        print(f"❌ TRAINING ERROR: {e}")
        training_status["status"] = "failed"
        training_status["message"] = str(e)

def evaluate_current_model():
    """
    Evaluates the model by comparing the 0.5 (median) quantile 
    against actual values.
    """
    if loader.tft_model is None:
        return {"error": "Model not trained or loaded"}

    try:
        # Use mode="raw" to get the full quantile output
        output = loader.tft_model.predict(loader.val_loader, mode="raw", return_x=False)
        
        # Calculate point predictions using the middle quantile (q=0.5)
        preds_list = []
        for target_pred in output.prediction:
            # target_pred shape: [Batch, Time, Quantiles]
            # Squeeze time and select the middle quantile index
            q50_idx = target_pred.shape[-1] // 2 
            preds_list.append(target_pred[:, 0, q50_idx])
            
        preds = torch.stack(preds_list, dim=-1).cpu().numpy()

        # Extract actuals from the dataloader
        actuals_list = []
        for batch in loader.val_loader:
            _, y = batch
            target_data = y[0] # The target values
            if isinstance(target_data, (list, tuple)):
                target_data = torch.stack(target_data, dim=-1)
            # Remove the time dimension for point-to-point comparison
            actuals_list.append(target_data.squeeze(1)) 
            
        actuals = torch.cat(actuals_list).cpu().numpy()

        # Calculate metrics for each specific target
        results = {}
        for i, col in enumerate(loader.targets):
            results[col] = {
                "MAE": round(float(mean_absolute_error(actuals[:, i], preds[:, i])), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(actuals[:, i], preds[:, i]))), 4)
            }
        return results

    except Exception as e:
        return {"error": f"Evaluation critical failure: {str(e)}"}