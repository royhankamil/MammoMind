# MammoMind : Breast Cancer Predictor

This project is a local web application for breast cancer prediction using ultrasound imaging, an affordable alternative to mammography and MRI.
It supports two input modes:

Numeric features (e.g., radius mean, texture mean, perimeter)

Ultrasound images

I developed the machine learning pipeline for numeric data and built deep learning models for image inputs using Python, TensorFlow, and PyTorch. We experimented with multiple models to achieve the highest accuracy.

The project was completed in 3–4 weeks in collaboration with a teammate.

## Instructions to Run the Breast Cancer Predictor Application
Prerequisites

Ensure that you have the following installed:
- Python 3.x
- Django (if not installed, you can install it using pip install django)

Steps to Run the Application
- Navigate to the project directory: Open the command prompt and navigate to the WebApp (Source/WebApp) directory by running the following command:     
     cd WebApp
- Start the Django development server: Once you're inside the WebApp directory, run the Django development server with the following command:
     python manage.py runserver

- Access the application: After running the server, open your web browser and go to the following URL to access the application:
    http://127.0.0.1:8000/

- manual installation, live preview:
     https://mammomind.azurewebsites.net/

     
data source : 
- https://www.kaggle.com/datasets/fatemehmehrparvar/breast-cancer-prediction
- https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer
- https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

