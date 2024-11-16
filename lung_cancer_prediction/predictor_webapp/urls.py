from django.urls import path
from . import views

urlpatterns = [
    path('formpredict/', views.text_based_predict, name='txtpredict'),  # Your home page with the form
    path('predictinput/', views.input_predict, name='predict_request'),  # The endpoint to handle the form submission
    path('predictimage/', views.image_predict, name='predictimage'),  # Updated this name
    path('imagepredict/', views.image_based_predict, name='imgpredict')
]
