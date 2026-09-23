from django.urls import path
from dashboard import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/stream/", views.video_stream, name="video_stream"),
    path("api/metrics/", views.api_metrics, name="api_metrics"),
    path("api/upload/", views.api_upload, name="api_upload"),
    path("api/datasets/", views.api_datasets, name="api_datasets"),
    path("api/control/", views.api_control, name="api_control"),
]
