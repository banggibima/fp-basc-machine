from fastapi import FastAPI, UploadFile, File
import uvicorn
from load_model import load_unet_model
from utils import process_image, segment_plate, perform_ocr

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Memuat model U-Net yang telah dilatih
model = load_unet_model("models/unet_license_plate_model.h5")

# Endpoint untuk mengenali teks pada pelat nomor
@app.post("/recognize")
async def recognize_plate(file: UploadFile = File(...)):
    image = await file.read()  # Membaca file gambar yang diunggah
    preprocessed_image = process_image(image)  # Memproses gambar
    segmented_plate = segment_plate(preprocessed_image, model)  # Segmentasi pelat nomor
    text = perform_ocr(segmented_plate)  # Mengenali teks pada pelat nomor
    return {"license_plate_text": text}  # Mengembalikan hasil pengenalan teks

# Menjalankan aplikasi jika file ini dieksekusi langsung
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
