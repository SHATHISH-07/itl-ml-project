import pandas as pd
from core import loader
from services.ml_service import predict_future_metrics
# Assuming llm_service has this function defined
from services.llm_service import generate_natural_language_summary 

def get_employee_profile(emp_id: str):
    """Utility to fetch basic info for the router"""
    with loader.resource_lock:
        if loader.df is None:
            return None
        emp_data = loader.df[loader.df.Employee_ID == emp_id]
        if emp_data.empty:
            return None
        
        last_row = emp_data.iloc[-1]
        return {
            "Employee_ID": last_row["Employee_ID"],
            "Department": last_row["Department"],
            "Role": last_row["Role"]
        }

def run_employee_analysis(emp_id: str, analysis_type: str, start_date=None, end_date=None):
    with loader.resource_lock:
        if loader.df is None:
            raise ValueError("Data not loaded in system")
            
        emp_df = loader.df[loader.df.Employee_ID == emp_id].copy()

        if emp_df.empty:
            raise ValueError("Employee not found")

        # Handle different analysis types
        if analysis_type == "overall_past":
            df_slice = emp_df.sort_values("Week").tail(30)
            raw_data = analyze_past_data(df_slice)
            sys_prompt = "Summarize the employee's historical performance."

        elif analysis_type == "custom_past":
            # Filter by date if provided
            emp_df['Week'] = pd.to_datetime(emp_df['Week'])
            mask = (emp_df['Week'] >= pd.to_datetime(start_date)) & (emp_df['Week'] <= pd.to_datetime(end_date))
            df_slice = emp_df.loc[mask]
            if df_slice.empty:
                raise ValueError("No data found for the selected date range")
            raw_data = analyze_past_data(df_slice)
            sys_prompt = f"Summarize performance between {start_date} and {end_date}."

        elif analysis_type == "forecast":
            raw_data = predict_future_metrics(emp_df)
            sys_prompt = "Summarize the 4-week performance forecast."

        else:
            raise ValueError(f"Invalid analysis type: {analysis_type}")

    summary = generate_natural_language_summary(sys_prompt, raw_data)

    return {
        "analysis_type": analysis_type,
        "raw_data": raw_data,
        "llm_summary": summary
    }

def analyze_past_data(df_slice):
    results = {}
    for col in loader.targets:
        if col in df_slice.columns:
            avg = df_slice[col].mean()
            results[col] = {"avg": round(float(avg), 2)}
    return results