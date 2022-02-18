from django.urls import path

from . import views


urlpatterns = [
    # POST
    path('v2/', views.view_pick_v2, name='view_pick_v2'),
    path('pick_cash_credit/', views.view_pick_cash_credit, name='view_pick_cash_credit'),
]
