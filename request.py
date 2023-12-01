# STROKE PREDICTION
import requests
import json

# api-endpoint
URL = "http://127.0.0.1:8000/stroke_prediction"

# defining a params dict for the parameters to be sent to the API
values = {
    "gender": "male",
    "age": 34,
    "hypertension": "yes",
    "heart_disease": "yes",
    "ever_married": "yes",
    "work_type": "private",
    "residence_type": "rural",
    "avg_glucose_level": 67,
    "bmi": 23,
    "smoking_status": "smokes",
    "get_bmi_times_glucose": 5,
    "age2_per_bmi": 2,
}
# sending get request and saving the response as response object
request = requests.post(url=URL, json=values)

print(request.text)
