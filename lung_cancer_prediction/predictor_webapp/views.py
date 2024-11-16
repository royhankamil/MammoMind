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

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


# Load the .h5 model for image prediction
model = tf.keras.models.load_model(r'predictor_webapp\breast-cancer-ultrasound.h5')

# Load the .pkl model for text-based prediction (using joblib)
pkl_model = joblib.load(r'predictor_webapp\breast_cancer_prediction.pkl')

def text_based_predict(request):
    return render(request, 'predict-form-text.html')

def image_based_predict(request):
    return render(request, 'predict-form-image.html')

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

            # Return the prediction result as JSON
            return JsonResponse({"prediction": prediction.tolist()})

        except Exception as e:
            print(f"Error processing the prediction request: {e}")
            return JsonResponse({"error": "An error occurred during prediction."}, status=400)

    else:
        return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=405)
    

def article(request):
    return render(request, 'article.html')