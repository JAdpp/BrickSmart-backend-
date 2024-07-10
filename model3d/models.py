from django.db import models
from django.contrib.auth.models import User

class ModelTask(models.Model):
    task_id = models.CharField(max_length=255, unique=True)
    prompt = models.TextField()
    bricks = models.CharField(max_length=255)  # Assuming a simple character field is sufficient
    model_download_url = models.URLField()
    image_download_url = models.URLField()
    lego_url = models.URLField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.task_id


