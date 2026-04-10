import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from core import loader
from services.training_service import training_status
import numpy as np

def predict_future_metrics(emp_df: pd.DataFrame) -> dict:
    from core import loader

    with loader.resource_lock:
        if loader.tft_model is None:
            raise RuntimeError("Model not loaded")

        if len(emp_df) < loader.max_encoder_length:
            raise ValueError("Not enough historical data for prediction")

        encoder_data = emp_df.tail(loader.max_encoder_length)
        last_row = encoder_data.iloc[-1]

        future = []

        for i in range(1, loader.max_prediction_length + 1):
            new = last_row.copy()
            new["time_idx"] += i
            new["Week"] += pd.Timedelta(days=7*i)

            # ✅ Dynamic future simulation
            for t in loader.targets:
                trend = encoder_data[t].iloc[-1] - encoder_data[t].iloc[-3]
                noise = np.random.normal(0, 0.5)
                new[t] = encoder_data[t].iloc[-1] + 0.3 * trend + noise

            new["lag_1_score"] = encoder_data["Project_Score"].iloc[-1]
            new["rolling_score"] = encoder_data["Project_Score"].mean()

            future.append(new)

        predict_df = pd.concat([encoder_data, pd.DataFrame(future)], ignore_index=True)

        dataset = TimeSeriesDataSet.from_dataset(loader.training_dataset, predict_df, predict=True)
        data_loader = dataset.to_dataloader(train=False, batch_size=1)

        preds = loader.tft_model.predict(data_loader)

        if isinstance(preds, list):
            preds = torch.stack(preds, dim=-1)

        preds = preds[0].cpu().numpy()

        # ✅ FIX: take median quantile
        preds = preds[..., 0]

    forecast_result = {}
    for i, col in enumerate(loader.targets):
        vals = preds[:, i]

        trend = (
            "Increase" if vals[-1] > vals[0] * 1.02
            else "Decrease" if vals[-1] < vals[0] * 0.98
            else "Stable"
        )

        forecast_result[col] = {
            "forecast": [round(float(v), 2) for v in vals],
            "trend": trend
        }

    return forecast_result