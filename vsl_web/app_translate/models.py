from django.db import models

class Vocab(models.Model):
    word = models.CharField(max_length=100, unique=True, verbose_name="Từ vựng")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.word
