# predictor/urls.py
from django.urls import path
from . import views  # Correct import from the predictor app

urlpatterns = [
    path("", views.home, name="home"), # http://127.0.0.1:8000/
    path('loan_approval/', views.loan_approval_view, name='loan_approval'),  # http://127.0.0.1:8000/predict/ Map the root to loan_approval_view
]
