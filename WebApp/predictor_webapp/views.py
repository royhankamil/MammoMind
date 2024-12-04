import json
import tensorflow as tf
import numpy as np
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
import os

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


# Load the .h5 model for image prediction
base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
h5_model_path = os.path.join(base_dir, 'breast-cancer-ultrasound.h5')
pkl_model_path = os.path.join(base_dir, 'breast_cancer_prediction.pkl')

# Load the .h5 model
try:
    model = tf.keras.models.load_model(h5_model_path)
    print("H5 model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading H5 model: {e}")

# Load the .pkl model
try:
    pkl_model = joblib.load(pkl_model_path)
    print("PKL model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading PKL model: {e}")

def text_based_predict(request):
    return render(request, 'predict-form-text.html')

def image_based_predict(request):
    return render(request, 'predict-form-image.html')

@csrf_exempt
def input_predict(request):
    if request.method == "POST":
        try:
            # Parse the request body (this assumes the data is in JSON format)
            data = json.loads(request.body)
            print("Raw data before parsing:", data)

            # List of numeric fields that are required for prediction
            required_fields = [
                'average_radius', 'average_texture', 'average_perimeter', 'average_area', 
                'average_smoothness', 'average_compactness', 'average_concavity', 'average_concave_points', 
                'average_symmetry', 'average_fractal_dimension', 'se_radius', 'se_texture', 'se_perimeter', 
                'se_area', 'se_smoothness', 'se_compactness', 'se_concavity', 'se_concave_points', 
                'se_symmetry', 'se_fractal_dimension', 'worst_radius', 'worst_texture', 'worst_perimeter', 
                'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points', 
                'worst_symmetry', 'worst_fractal_dimension'
            ]
            
            # Safely parse the data and handle potential conversion issues
            parsed_data = {}
            for key, value in data.items():
                if key in required_fields:  # Only process the required fields
                    try:
                        parsed_data[key] = float(value.strip()) if isinstance(value, str) and value.strip() else float(value) if value else 0.0
                    except Exception as e:
                        print(f"Error parsing key '{key}' with value '{value}': {e}")
                        parsed_data[key] = 0.0  # Default to 0.0 if conversion fails

            print("Parsed data:", parsed_data)

            # Prepare input data for the model
            input_array = np.array([[parsed_data[field] for field in required_fields]])

            # Make the prediction using the .pkl model
            prediction_model = pkl_model.predict(input_array)
            if prediction_model == 0:
                prediction = 'Tidak terdeteksi adanya indikasi kanker payudara'
            else:
                prediction = 'Terdeteksi adanya indikasi kanker payudara'


            # Return the prediction result as JSON
            return JsonResponse({"prediction": prediction})

        except Exception as e:
            print(f"Error processing the prediction request: {e}")
            return JsonResponse({"error": "An error occurred during prediction."}, status=400)

    else:
        return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)


@csrf_exempt
def image_predict(request):
    if request.method == "POST":
        try:
            # Check if the file is in the request
            if 'patient_image' not in request.FILES:
                return JsonResponse({"error": "No image provided."}, status=400)

            # Get the uploaded file
            uploaded_file = request.FILES['patient_image']
            
            # Open the image using PIL
            img = Image.open(uploaded_file)
            
            # Convert to RGB if the image has an alpha channel (RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image to 150x150 as expected by the model
            img = img.resize((150, 150))  # Adjust the size to 150x150

            img_array = np.array(img) / 255.0  # Normalize the image data to [0, 1]
            
            # Preprocess the image as required by the model (e.g., VGG16)
            img_array = preprocess_input(img_array)  # Preprocess for VGG16
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make the prediction using the model
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)  # Get the class with the highest probability

            # Map predicted class to labels
            class_labels = ['benign', 'malignant', 'normal']
            predicted_label = class_labels[predicted_class]

            # Determine the prediction message
            if predicted_label in ['benign', 'normal']:
                prediction_message = 'Tidak terdeteksi adanya indikasi kanker payudara.'
            elif predicted_label == 'malignant':
                prediction_message = 'Terdeteksi adanya indikasi kanker payudara.'
            else:
                prediction_message = 'Hasil tidak jelas, silakan konsultasi lebih lanjut.'

            # Return the prediction result as JSON
            return JsonResponse({"prediction": prediction_message, "class": predicted_label})

        except Exception as e:
            print(f"Error processing the prediction request: {e}")
            return JsonResponse({"error": "An error occurred during prediction."}, status=400)

    else:
        return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)
   

def article(request):
    return render(request, 'article.html')