import tensorflow as tf
from tensorflow import keras
import numpy as np
from efficientnet.tfkeras import preprocess_input
from PIL import Image
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('float32')

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

class CustomInputLayer(keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]  # remove batch size dimension
        super().__init__(*args, **kwargs)

model = keras.models.load_model(
    'Model/final_model.h5',
    custom_objects={
        'InputLayer': CustomInputLayer
        
        }
)


print("loaded")
# Preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((260, 260))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  
    return img_array

# Function to get prediction
def predict_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASSES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities: {predictions[0]}")

