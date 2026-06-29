from django.contrib import admin
from .models import Vocab

@admin.register(Vocab)
class VocabAdmin(admin.ModelAdmin):
    list_display = ('word', 'created_at')
    search_fields = ('word',)
