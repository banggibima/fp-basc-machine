from fastapi import FastAPI, UploadFile, File
import uvicorn
from load_model import load_unet_model
from utils import process_image, segment_plate, perform_ocr

app = FastAPI()

# Load the trained U-Net model
model = load_unet_model("models/unet_license_plate_model.h5")

@app.post("/recognize")
async def recognize_plate(file: UploadFile = File(...)):
    image = await file.read()
    preprocessed_image = process_image(image)
    segmented_plate = segment_plate(preprocessed_image, model)
    text = perform_ocr(segmented_plate)
    return {"license_plate_text": text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
