# uvicorn main:app --reload
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException
from classes1 import *
from functions import *

app = FastAPI()

stroke_predictor = joblib.load("stroke_classifier.joblib")


@app.get("/")
def home():
    return {"text": "Health predictions"}


@app.post("/stroke_prediction")
async def create_application(stroke_pred: stroke_prediction):
    stroke_df = pd.DataFrame()

    if stroke_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="Gender not found")

    if stroke_pred.hypertension not in hypertension_dict:
        raise HTTPException(status_code=404, detail="Hypertension disease not defined")

    if stroke_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")

    if stroke_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")

    if stroke_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")

    if stroke_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")

    if stroke_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")

    stroke_df["gender"] = [stroke_pred.gender]
    stroke_df["age"] = [stroke_pred.age]
    stroke_df["hypertension"] = [stroke_pred.hypertension]
    stroke_df["heart_disease"] = [stroke_pred.heart_disease]
    stroke_df["ever_married"] = [stroke_pred.ever_married]
    stroke_df["work_type"] = [stroke_pred.work_type]
    stroke_df["residence_type"] = [stroke_pred.residence_type]
    stroke_df["avg_glucose_level"] = [stroke_pred.avg_glucose_level]
    stroke_df["bmi"] = [stroke_pred.bmi]
    stroke_df["smoking_status"] = [stroke_pred.smoking_status]
    stroke_df["get_bmi_times_glucose"] = [stroke_pred.get_bmi_times_glucose]
    stroke_df["age2_per_bmi"] = [stroke_pred.age2_per_bmi]

    prediction = stroke_predictor.predict(stroke_df)
    if prediction[0] == 0:
        prediction = "You are not likely to get a stroke."
    else:
        prediction = "You are on a risk to get a stroke."

    return {"prediction": prediction}
