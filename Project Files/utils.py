import os
from ibm_watson_machine_learning.foundation_models import Model
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def init_granite_model():
    """
    Initializes and returns the IBM Granite 13B Instruct v2 model.
    """
    api_key = os.getenv("IBM_WATSON_ML_API_KEY")
    url = os.getenv("IBM_WATSON_ML_URL")
    project_id = os.getenv("IBM_WATSON_ML_PROJECT_ID")

    if not api_key or not url or not project_id:
        raise ValueError("IBM_WATSON_ML_API_KEY, IBM_WATSON_ML_URL, and IBM_WATSON_ML_PROJECT_ID must be set in .env file.")

    model_id = "ibm/granite-13b-instruct-v2"

    gen_parms = {
        "max_new_tokens": 500,
        "min_new_tokens": 50,
        "temperature": 0.2,
        "repetition_penalty": 1.0
    }

    model = Model(
        model_id=model_id,
        credentials={"apikey": api_key, "url": url},
        project_id=project_id,
        params=gen_parms
    )
    return model

def get_sample_patient_data():
    """
    Generates and returns sample patient health metrics.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    data = []
    start_date = datetime.now() - timedelta(days=30)
    for i in range(30):
        date = start_date + timedelta(days=i)
        heart_rate = int(np.random.normal(70, 5))
        systolic_bp = int(np.random.normal(120, 8))
        diastolic_bp = int(np.random.normal(80, 5))
        blood_glucose = int(np.random.normal(95, 10))
        data.append([date, heart_rate, systolic_bp, diastolic_bp, blood_glucose])

    df = pd.DataFrame(data, columns=['Date', 'Heart Rate', 'Systolic BP', 'Diastolic BP', 'Blood Glucose'])
    return df

def get_patient_profile():
    """
    Returns a sample patient profile.
    """
    return {
        "age": 35,
        "gender": "Female",
        "medical_history": "No significant medical history, occasional seasonal allergies."
    }