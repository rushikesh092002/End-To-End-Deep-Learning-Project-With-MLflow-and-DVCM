import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Load model once during initialization
        self.model = load_model(os.path.join("artifacts", "training", "model.h5"))


    def predict(self):
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # optional: normalize if required

        result = np.argmax(self.model.predict(test_image, verbose=0), axis=1)

        prediction = 'Tumor' if result[0] == 1 else 'Normal'
        return [{ "image": prediction }]
