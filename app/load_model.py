from keras.models import load_model

def load_unet_model(model_path):
    return load_model(model_path)
