# app/main.py

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import torch
import torch.nn as nn
import cv2
import numpy as np
from io import BytesIO
from fastapi.responses import JSONResponse
from typing import List
from .utils import preprocess_image, preprocess_tabular
from .model import MultimodalFusionNet

# Create FastAPI app
app = FastAPI()

# Load the pre-trained models
image_model = torch.load('models/image_model.pth', map_location=torch.device('cpu'))
image_model.eval()

tabular_model = torch.load('models/tabular_model.pth', map_location=torch.device('cpu'))
tabular_model.eval()

# Multimodal Fusion Model
fusion_model = MultimodalFusionNet(image_model, tabular_model)

# Pydantic Model for Tabular Data Input
class TabularData(BaseModel):
    age: int
    sex: str
    bmi: float
    # Add other relevant tabular features

# Endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...), tabular_data: TabularData = Form(...)):
    image_bytes = await file.read()
    tabular_input = tabular_data.dict()
    
    # Preprocess the image
    img = preprocess_image(image_bytes)
    
    # Preprocess the tabular data
    tabular_input = preprocess_tabular(tabular_input)
    
    # Model inference
    with torch.no_grad():
        output = fusion_model(img, tabular_input)
        prediction = output.argmax(dim=1).item()
    
    return JSONResponse(content={"prediction": prediction})

