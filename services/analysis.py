import pandas as pd
from core import loader
from services.ml_service import predict_future_metrics
from services.llm_service import generate_natural_language_summary
def run_employee_analysis(emp_id: str, analysis_type: str):
    from core import loader

    with loader.resource_lock:
        emp_df = loader.df[loader.df.Employee_ID == emp_id].copy()

        if emp_df.empty:
            raise ValueError("Employee not found")

        if analysis_type == "overall_past":
            df_slice = emp_df.sort_values("Week").tail(30)
            raw_data = analyze_past_data(df_slice)
            sys_prompt = "Summarize the employee's historical performance."

        elif analysis_type == "forecast":
            raw_data = predict_future_metrics(emp_df)
            sys_prompt = "Summarize the 4-week performance forecast."

        else:
            raise ValueError("Invalid analysis type")

    summary = generate_natural_language_summary(sys_prompt, raw_data)

    return {
        "analysis_type": analysis_type,
        "raw_data": raw_data,
        "llm_summary": summary
    }

def analyze_past_data(df_slice):
    results = {}
    for col in loader.targets:
        avg = df_slice[col].mean()
        results[col] = {"avg": round(float(avg), 2)}
    return results