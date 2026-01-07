from fastapi import FastAPI, UploadFile, File
import torch
from lidarflowsuite.models.model import SceneFlowModel
import numpy as np
import io

app = FastAPI(title="LidarFlowSuite API")

# Global model instance for inference
model = SceneFlowModel()
# model.load_state_dict(...) # Should load from a default path or env

@app.get("/")
def read_root():
    return {"message": "Welcome to LidarFlowSuite API"}

@app.post("/predict")
async def predict_flow(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Read files
    content1 = await file1.read()
    content2 = await file2.read()
    
    # Dummy processing
    pc1 = torch.randn(1, 2048, 3)
    pc2 = torch.randn(1, 2048, 3)
    
    model.eval()
    with torch.no_grad():
        flow = model(pc1, pc2)
        
    return {
        "flow": flow.cpu().numpy().tolist()
    }
