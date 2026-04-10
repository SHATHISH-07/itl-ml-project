import pandas as pd
import numpy as np
import torch
import os
import datetime
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import MultiNormalizer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core import loader

training_status = {
    "status": "idle",
    "progress": 0,
    "last_error": None,
    "last_trained": None,
    "message": "Ready"
}

# =========================
# ✅ PREPROCESS DATA
# =========================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Week"] = pd.to_datetime(df["Week"])
    df = df.sort_values(["Employee_ID", "Week"])

    df["time_idx"] = df.groupby("Employee_ID").cumcount()
    df["Employee_ID_encoded"] = df["Employee_ID"].astype("category").cat.codes

    df["lag_1_score"] = df.groupby("Employee_ID")["Project_Score"].shift(1)
    df["lag_2_score"] = df.groupby("Employee_ID")["Project_Score"].shift(2)
    df["lag_3_score"] = df.groupby("Employee_ID")["Project_Score"].shift(3)

    df["lag_1_hours"] = df.groupby("Employee_ID")["Hours_Worked"].shift(1)
    df["lag_2_hours"] = df.groupby("Employee_ID")["Hours_Worked"].shift(2)

    df["rolling_score"] = df.groupby("Employee_ID")["Project_Score"].rolling(3).mean().reset_index(0, drop=True)
    df["rolling_hours"] = df.groupby("Employee_ID")["Hours_Worked"].rolling(3).mean().reset_index(0, drop=True)

    df["task_pressure"] = df["Tasks_Completed"] / (df["Hours_Worked"] + 1)

    df["score_change"] = df.groupby("Employee_ID")["Project_Score"].diff()
    df["hours_change"] = df.groupby("Employee_ID")["Hours_Worked"].diff()

    df["month"] = df["Week"].dt.month
    df["weekofyear"] = df["Week"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Week"].dt.dayofweek

    df = df.bfill().ffill()

    return df


# =========================
# ✅ CREATE DATASETS
# =========================
def create_datasets(df: pd.DataFrame):
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

        time_varying_unknown_reals=[],  # no leakage

        time_varying_known_reals=[
            "time_idx", "lag_1_score", "lag_2_score", "lag_3_score",
            "lag_1_hours", "lag_2_hours",
            "rolling_score", "rolling_hours",
            "task_pressure", "score_change", "hours_change",
            "month", "weekofyear", "day_of_week"
        ],

        target_normalizer=MultiNormalizer([
            GroupNormalizer(groups=["Employee_ID_encoded"])
            for _ in loader.targets
        ]),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=2)
    val_loader = validation.to_dataloader(train=False, batch_size=32, num_workers=2)

    return training, validation, train_loader, val_loader


# =========================
# ✅ TRAINING PIPELINE
# =========================
def run_training_pipeline():
    global training_status

    try:
        training_status.update({"status": "training", "progress": 10, "message": "Preprocessing..."})

        df = pd.read_json("data/employee_data.json")
        df = preprocess_data(df)

        training, validation, train_loader, val_loader = create_datasets(df)

        training_status.update({"progress": 30, "message": "Building model..."})

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.003,
            hidden_size=128,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            loss=QuantileLoss(),
            output_size=[len(QuantileLoss().quantiles)] * len(loader.targets),
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath="model",
            filename="best_tft",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=40,
            accelerator="auto",
            gradient_clip_val=0.1,
            callbacks=[checkpoint_callback, early_stop],
            logger=False
        )

        training_status.update({"progress": 60, "message": "Training..."})

        trainer.fit(tft, train_loader, val_loader)

        # ✅ LOAD BEST MODEL
        best_model_path = checkpoint_callback.best_model_path
        print("Best model:", best_model_path)

        if best_model_path:
            loader.tft_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
            loader.tft_model.eval()

            # 🔥 CRITICAL FIX
            loader.val_loader = val_loader
            loader.training_dataset = training

        training_status.update({
            "status": "completed",
            "progress": 100,
            "message": "Training complete!",
            "last_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        training_status.update({
            "status": "failed",
            "last_error": str(e),
            "message": "Training failed"
        })


# =========================
# ✅ EVALUATION
# =========================
def evaluate_current_model() -> dict:

    if loader.tft_model is None:
        return {"error": "Model not trained"}

    if loader.val_loader is None:
        return {"error": "Validation loader missing"}

    try:
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

        # ✅ REMOVE QUANTILE DIM
        preds = preds[..., 0]

        # ✅ ALIGN LENGTH
        min_len = min(actuals.shape[0], preds.shape[0])
        actuals = actuals[:min_len]
        preds = preds[:min_len]

        # ✅ FLATTEN
        actuals = actuals.reshape(-1, actuals.shape[-1])
        preds = preds.reshape(-1, preds.shape[-1])

        results = {}

        for i, col in enumerate(loader.targets):
            mae = mean_absolute_error(actuals[:, i], preds[:, i])
            rmse = np.sqrt(mean_squared_error(actuals[:, i], preds[:, i]))

            if np.var(actuals[:, i]) == 0:
                r2 = 1.0 if mae == 0 else 0.0
            else:
                r2 = r2_score(actuals[:, i], preds[:, i])

            results[col] = {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4)
            }

        return results

    except Exception as e:
        return {"error": str(e)}