from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('', include('display.urls'), name='display'),
    path('api/', include('api.urls'), name='api'),

    path('admin/', admin.site.urls),
]
