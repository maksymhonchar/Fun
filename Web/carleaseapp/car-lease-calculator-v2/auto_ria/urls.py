from django.urls import path

from . import views


urlpatterns = [
    path('marks/', views.view_marks, name='view_marks'),
    path('models/', views.view_models, name='view_models'),

    path('average_price/', views.view_average_price, name='view_average_price'),
]
