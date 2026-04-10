from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import shutil
import os
from services import training_service

router = APIRouter(prefix="/api/model", tags=["Model Training & Evaluation"])

@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. Check if already training
    if training_service.training_status["status"] == "training":
        raise HTTPException(status_code=400, detail="A training process is already running.")

    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    
    os.makedirs("data", exist_ok=True)
    file_location = "data/employee_data.json"
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    background_tasks.add_task(training_service.run_training_pipeline)
    
    return {"message": "Model training started."}

@router.get("/status")
async def get_training_status():
    return training_service.training_status

@router.get("/evaluate")
async def evaluate_model():
    results = training_service.evaluate_current_model()
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    return results