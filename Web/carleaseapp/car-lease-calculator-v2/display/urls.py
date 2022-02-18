from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='view_index'),

    path('car/', views.car, name='view_car'),
    path('taxi/', views.taxi, name='view_taxi'),
    path('cash/', views.cash, name='view_cash'),

    path('ir/', views.ir_all, name='view_ir_all'),
    path('ir/new/', views.ir_new, name='view_ir_new'),
    path('ir/<int:ir_calculation_id>/', views.ir_one, name='view_ir_one'),
]
