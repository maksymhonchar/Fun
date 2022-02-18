from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    # Frontend
    path('', include('display.urls'), name='display'),

    # Admin page
    path('admin/', admin.site.urls),

    # API
    path('api/exchange_rates/', include('exchange_rates.urls'), name='exchange_rates'),
    path('api/auto_ria/', include('auto_ria.urls'), name='auto_ria'),
    path('api/fin_picking/', include('fin_picking.urls'), name='fin_picking'),
    path('api/ir_picking/', include('ir_picking.urls'), name='ir_picking'),
]
