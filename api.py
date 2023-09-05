import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

app = Flask(__name__)

MODEL = "resnet.pkl"

# Load the pre-trained VGG16 model (replace with your actual model file)
model = pickle.load(MODEL)

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        image_file = request.files['image']
        
        # Check if the file is None
        if image_file is None:
            return jsonify({'error': 'No image provided'}), 400

        # Read the image file
        img = image.load_img(image_file, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        # Make predictions
        predictions = model.predict(img)
        
        # Decode the predictions (replace with your own label decoding logic)
        class_names = ['Abrasions','Bruises','Burns','Cut','Ingrown_nails','Laceration','Stab_wound']  # Replace with your class names
        predicted_class = class_names[np.argmax(predictions)]
        
        return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
