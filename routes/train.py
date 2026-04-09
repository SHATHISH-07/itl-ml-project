from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import shutil
import os
from services import training_service

router = APIRouter(prefix="/api/model", tags=["Model Training & Evaluation"])

@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads new employee data and starts model training in the background."""
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported.")
    
    os.makedirs("data", exist_ok=True)
    file_location = "data/employee_data.json"
    
    # Save the uploaded file
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    # Trigger background task so the API doesn't hang
    background_tasks.add_task(training_service.run_training_pipeline)
    
    return {
        "message": "Data uploaded successfully. Model training has started in the background. Check server console for progress."
    }

@router.get("/evaluate")
async def evaluate_model():
    """Returns the MAE, RMSE, and R2 score of the currently loaded model."""
    results = training_service.evaluate_current_model()
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    return results