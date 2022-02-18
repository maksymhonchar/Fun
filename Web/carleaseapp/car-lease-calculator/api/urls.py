from django.urls import path

from . import views

urlpatterns = [
    path('rates/', views.view_rates, name='rates'),
    path('rates/<str:currency_code>/', views.view_rates_currency_code, name='rates_currency_code'),

    path('pick/find_rate_having_irr/', views.view_pick_find_rate_having_irr, name='pick_find_rate_having_irr'),
]
