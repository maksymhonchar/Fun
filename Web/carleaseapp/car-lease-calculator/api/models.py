import datetime

from django.db import models


class CurrencyRateRequest(models.Model):
    create_time = models.DateTimeField(default=datetime.datetime.now, blank=False, null=False)
    source = models.CharField(max_length=100, blank=False, null=False)
    payload = models.TextField()
