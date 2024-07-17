from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import generate_model, generate_model_image

urlpatterns = [
    path("model/", generate_model, name='generate_model'),
    path("model/image/", generate_model_image, name='generate_model_image'),
]

# 在开发环境下添加媒体文件的服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)