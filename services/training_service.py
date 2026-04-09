import pandas as pd
import numpy as np
import torch
import os
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core import loader

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Exact feature engineering from your Colab notebook."""
    df["Week"] = pd.to_datetime(df["Week"])
    df = df.sort_values(["Employee_ID", "Week"])

    df["time_idx"] = df.groupby("Employee_ID").cumcount()
    df["Employee_ID_encoded"] = df["Employee_ID"].astype("category").cat.codes

    df["lag_1_score"] = df.groupby("Employee_ID")["Project_Score"].shift(1)
    df["rolling_score"] = df.groupby("Employee_ID")["Project_Score"].rolling(3).mean().reset_index(0, drop=True)
    df["lag_2_score"] = df.groupby("Employee_ID")["Project_Score"].shift(2)
    df["lag_3_score"] = df.groupby("Employee_ID")["Project_Score"].shift(3)
    df["lag_1_hours"] = df.groupby("Employee_ID")["Hours_Worked"].shift(1)
    df["lag_2_hours"] = df.groupby("Employee_ID")["Hours_Worked"].shift(2)

    df["task_pressure"] = df["Tasks_Completed"] / (df["Hours_Worked"] + 1)
    df["high_workload"] = (df["Workload_Level"] > 7).astype(int)

    df["score_change"] = df.groupby("Employee_ID")["Project_Score"].diff()
    df["hours_change"] = df.groupby("Employee_ID")["Hours_Worked"].diff()

    df["month"] = df["Week"].dt.month
    df["weekofyear"] = df["Week"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Week"].dt.dayofweek

    df["rolling_hours"] = df.groupby("Employee_ID")["Hours_Worked"].rolling(3).mean().reset_index(0, drop=True)
    
    df = df.bfill().ffill()
    return df

def create_datasets(df: pd.DataFrame):
    """Creates the training and validation loaders."""
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
            "time_idx", "lag_1_score", "lag_2_score", "lag_3_score",
            "lag_1_hours", "lag_2_hours", "rolling_score", "rolling_hours",
            "task_pressure", "score_change", "hours_change", "month",
            "weekofyear", "day_of_week"
        ],
        target_normalizer=MultiNormalizer([GroupNormalizer(groups=["Employee_ID_encoded"]) for _ in loader.targets]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    train_loader = training.to_dataloader(train=True, batch_size=32)
    val_loader = validation.to_dataloader(train=False, batch_size=32)
    
    return training, validation, train_loader, val_loader

def run_training_pipeline():
    """Trains the model in the background and hot-reloads it into the app."""
    print("--- Starting Background Training Process ---")
    
    df = pd.read_json("data/employee_data.json")
    df = preprocess_data(df)
    training, validation, train_loader, val_loader = create_datasets(df)

    loss_function = QuantileLoss()
    n_targets = len(loader.targets)
    n_quantiles = len(loss_function.quantiles)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.005,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=16,
        loss=loss_function,
        output_size=[n_quantiles] * n_targets,
    )

    os.makedirs("model", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="model",
        filename="tft_model", 
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=5)],
        gradient_clip_val=0.1,
    )

    trainer.fit(tft, train_loader, val_loader)
    print("Training Complete. Reloading server state...")
    
    # Reload the new model and data into the live server
    loader.reload_data_and_tft()
    print("Server successfully updated with new model!")

def evaluate_current_model() -> dict:
    """Exact Colab evaluation logic for MAE, RMSE, and R2."""
    if not loader.tft_model or not loader.val_loader:
        return {"error": "Model or validation data not loaded."}

    actuals_list = []
    for batch in loader.val_loader:
        x, y = batch
        target = y[0]
        if isinstance(target, (list, tuple)):
            target = torch.stack(list(target), dim=-1)
        actuals_list.append(target)

    actuals = torch.cat(actuals_list).cpu().numpy()
    preds = loader.tft_model.predict(loader.val_loader)

    if isinstance(preds, (list, tuple)):
        preds = torch.stack(list(preds), dim=-1)
    preds = preds.cpu().numpy()

    actuals = actuals.reshape(-1, len(loader.targets))
    preds = preds.reshape(-1, len(loader.targets))

    results = {}
    for i, col in enumerate(loader.targets):
        mae = mean_absolute_error(actuals[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(actuals[:, i], preds[:, i]))
        r2 = r2_score(actuals[:, i], preds[:, i])
        results[col] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}

    return results