from django.urls import path

from . import views


urlpatterns = [
    # GET
    path('calculations/', views.view_ir_calculations, name='view_ir_calculations'),
    path('calculation/', views.view_ir_calculation, name='view_ir_calculation'),

    # POST
    path('new_calculation/', views.view_new_calculation, name='view_new_calculation'),
]
