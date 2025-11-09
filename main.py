from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("crop_model_temp_humidity_soil.pkl")
label_encoder_soil = joblib.load("label_encoder_soil.pkl")
label_encoder_crop = joblib.load("label_encoder_crop_temp_humidity_soil.pkl")

app = FastAPI(title="🌿 Crop Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CropInput(BaseModel):
    Temparature: float
    Humidity: float
    Soil_Type: str

@app.get("/")
def read_root():
    return {"message": "🌱 Crop Prediction API is running!"}

@app.post("/predict")
def predict_crop(data: CropInput):
    try:
        soil_encoded = label_encoder_soil.transform([data.Soil_Type])[0]

        sample = pd.DataFrame(
            [[data.Temparature, data.Humidity, soil_encoded]],
            columns=['Temparature', 'Humidity', 'Soil Type']
        )

        predicted_crop = label_encoder_crop.inverse_transform(model.predict(sample))[0]

        return {
            "Predicted Crop": predicted_crop,
            "Input": {
                "Temparature": data.Temparature,
                "Humidity": data.Humidity,
                "Soil Type": data.Soil_Type
            }
        }
    except Exception as e:
        return {"error": str(e)}

