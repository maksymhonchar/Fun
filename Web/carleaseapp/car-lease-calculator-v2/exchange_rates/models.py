from django.db import models
from django.utils import timezone


class CurrencyRateRequest(models.Model):
    create_time = models.DateTimeField(default=timezone.now, blank=False)
    source = models.CharField(max_length=255, blank=False)
    payload = models.TextField()

    def __str__(self):
        return f'CurrencyRateRequest: [id] {self.id}, [create_time] {self.create_time}, [source] {self.source}, [payload] {self.payload}'
