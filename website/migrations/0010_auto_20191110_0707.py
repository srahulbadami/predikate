# Generated by Django 2.2.7 on 2019-11-10 07:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0009_auto_20191109_1338'),
    ]

    operations = [
        migrations.AddField(
            model_name='custommodels',
            name='accuracy',
            field=models.IntegerField(default=170),
        ),
        migrations.AddField(
            model_name='custommodels',
            name='pre_model',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
