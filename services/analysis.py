import pandas as pd
from core import loader
from services import ml_service, llm_service

def get_employee_profile(identifier: str) -> dict:
    identifier = str(identifier).strip().lower()
    emp = loader.df[loader.df["Employee_ID"].str.lower() == identifier]
    if emp.empty:
        emp = loader.df[loader.df["Employee_Name"].str.lower() == identifier]
    
    if emp.empty:
        return None
    
    emp = emp.iloc[0]
    return {
        "Employee_ID": emp["Employee_ID"],
        "Employee_Name": emp["Employee_Name"],
        "Department": emp["Department"],
        "Role": emp["Role"]
    }

def analyze_past_data(df_slice: pd.DataFrame) -> dict:
    summary = {}
    for col in loader.targets:
        values = df_slice[col]
        mid = len(values) // 2
        first_half = values.iloc[:mid].mean()
        second_half = values.iloc[mid:].mean()
        trend = "Increasing" if second_half > first_half * 1.02 else ("Decreasing" if second_half < first_half * 0.98 else "Stable")
        summary[col] = {
            "avg": round(float(values.mean()), 2),
            "trend": trend
        }
    return summary

def run_employee_analysis(emp_id: str, analysis_type: str, start_date: str = None, end_date: str = None):
    emp_df = loader.df[loader.df.Employee_ID == emp_id].copy()
    
    if analysis_type == "overall_past":
        df_slice = emp_df.sort_values("Week").tail(30)
        raw_data = analyze_past_data(df_slice)
        sys_prompt = "You are an AI HR assistant. Review the past performance trends of this employee and write a short natural language summary highlighting strengths and weaknesses."
        
    elif analysis_type == "custom_past":
        df_slice = emp_df[
            (emp_df["Week"] >= pd.to_datetime(start_date)) & 
            (emp_df["Week"] <= pd.to_datetime(end_date))
        ]
        if df_slice.empty:
            raise ValueError("No data found for the selected date range.")
        raw_data = analyze_past_data(df_slice)
        sys_prompt = f"You are an AI HR assistant. Review the performance trends from {start_date} to {end_date} and summarize them."

    elif analysis_type == "forecast":
        raw_data = ml_service.predict_future_metrics(emp_df)
        sys_prompt = "You are an AI HR assistant. Explain the 4-week future forecast data provided to you in natural language. Highlight what metrics will increase or decrease."
    else:
        raise ValueError("Invalid analysis type")

    llm_summary = llm_service.generate_natural_language_summary(sys_prompt, raw_data)
    
    return {
        "analysis_type": analysis_type,
        "raw_data": raw_data,
        "llm_summary": llm_summary
    }