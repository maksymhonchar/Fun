from django.db import models
from django.utils import timezone


class IRCalculation(models.Model):
    create_time = models.DateTimeField(default=timezone.now, blank=False)
    payload = models.TextField()

    def __str__(self):
        return f'IRCalculation: [id] {self.id}, [create_time] {self.create_time}, [payload] {self.payload}'
