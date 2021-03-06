# Generated by Django 2.2.7 on 2019-11-11 10:51

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0016_auto_20191110_1131'),
    ]

    operations = [
        migrations.AddField(
            model_name='custommodels',
            name='model_used',
            field=models.CharField(default='RandomForestClassifier', max_length=255),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='custommodels',
            name='cus_model',
            field=models.FileField(upload_to='data/models/%Y/%m/%d/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['csv', 'tsv'])]),
        ),
        migrations.AlterField(
            model_name='custommodels',
            name='temp_data',
            field=models.FileField(upload_to='data/tempdata/%Y/%m/%d/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['csv', 'tsv'])]),
        ),
        migrations.AlterField(
            model_name='custommodels',
            name='upload',
            field=models.FileField(upload_to='data/datasets/%Y/%m/%d/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['csv', 'tsv'])]),
        ),
    ]
