from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import shutil, os
from services import training_service

router = APIRouter(prefix="/api/model")

import threading

@router.post("/train")
async def train(file: UploadFile = File(...)):

    if training_service.training_status["status"] == "training":
        raise HTTPException(400, "Training already running")

    os.makedirs("data", exist_ok=True)

    with open("data/employee_data.json", "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ✅ RUN TRAINING IN THREAD
    thread = threading.Thread(target=training_service.run_training_pipeline)
    thread.start()

    return {"message": "Training started in background"}

@router.get("/evaluate")
async def evaluate():
    res = training_service.evaluate_current_model()

    if "error" in res:
        raise HTTPException(400, res["error"])

    return res

@router.get("/status")
async def status():
    return training_service.training_status