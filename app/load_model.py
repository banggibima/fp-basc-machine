from keras.models import load_model

# Fungsi untuk memuat model U-Net dari path file yang diberikan
def load_unet_model(model_path):
    return load_model(model_path)   # Memuat model U-Net dari file
