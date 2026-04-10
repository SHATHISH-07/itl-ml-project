from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services import analysis

router = APIRouter(prefix="/api/analytics", tags=["Employee Analytics"])

class QueryRequest(BaseModel):
    employee_id: str
    user_query: Optional[str] = None
    analysis_type: str 
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@router.get("/employee/{employee_id}")
async def get_employee(employee_id: str):
    profile = analysis.get_employee_profile(employee_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Employee not found")
    return profile

@router.post("/query")
async def process_query(request: QueryRequest):
    profile = analysis.get_employee_profile(request.employee_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    if request.analysis_type == "custom_past":
        if not request.start_date or not request.end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date and end_date are required for custom_past"
        )
        
    try:
        result = analysis.run_employee_analysis(
            emp_id=profile["Employee_ID"],
            analysis_type=request.analysis_type,
            start_date=request.start_date,
            end_date=request.end_date
        )
        return {"employee": profile, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")