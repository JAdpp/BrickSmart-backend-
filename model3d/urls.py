from django.urls import path
from .views import generate_model, generate_model_image

urlpatterns = [
    path("model/", generate_model, name='generate_model'),
    path("model/image/", generate_model_image, name='generate_model_image'),
]