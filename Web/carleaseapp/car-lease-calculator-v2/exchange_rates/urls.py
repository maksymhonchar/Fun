from django.urls import path

from . import views


urlpatterns = [
    path('nbu/', views.view_nbu, name='view_nbu'),
]
