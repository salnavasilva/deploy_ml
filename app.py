import io

import joblib
import numpy as np
from datasets import load_dataset
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from deploy_ml.preprocessing.data_preprocessor import DataPreprocess

app = FastAPI()

# Load the model artifact
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# We need the encoder to process the input image into embeddings
# Since the encoder is part of DataPreprocess, we might need to instantiate it
# or refactor DataPreprocess to allow standalone encoder usage.
# For now, let's assume we can use the same logic.
encoder = None

@app.on_event("startup")
def load_encoder():
    global encoder
    from embetter.multi import ClipEncoder
    encoder = ClipEncoder()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)
    
    try:
        request_object_content = await file.read()
        img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
        
        # The encoder.transform expects a list of images
        embeddings = encoder.transform([img])[0]
        
        # Ensure embeddings are 2D for scikit-learn
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        prediction = model.predict(embeddings)[0]
        
        # Get probabilities for confidence score
        probabilities = model.predict_proba(embeddings)[0]
        confidence = float(np.max(probabilities))
        
        # Map prediction to label
        # Assuming 0 is cat, 1 is dog based on dataset
        label = "dog" if prediction == 1 else "cat"
        
        return JSONResponse(content={
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {
        "message": "Model deployment is running!",
        "endpoint": "/predict",
        "instructions": "Send a POST request to /predict with an image file in the body."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
