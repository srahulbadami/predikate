# Generated by Django 2.2.7 on 2019-11-10 11:31

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0015_custommodels_temp_data'),
    ]

    operations = [
        migrations.AlterField(
            model_name='custommodels',
            name='cus_model',
            field=models.FileField(upload_to='models/%Y/%m/%d/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['csv', 'tsv'])]),
        ),
        migrations.AlterField(
            model_name='custommodels',
            name='temp_data',
            field=models.FileField(upload_to='tempdata/%Y/%m/%d/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['csv', 'tsv'])]),
        ),
    ]
