# Generated by Django 2.2.7 on 2019-11-10 08:34

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0013_auto_20191110_0828'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='custommodels',
            name='pre_model',
        ),
    ]
