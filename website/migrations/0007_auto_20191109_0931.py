# Generated by Django 2.2.7 on 2019-11-09 09:31

from django.db import migrations, models
import website.validators


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0006_custommodels_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='custommodels',
            name='upload',
            field=models.FileField(upload_to='datasets/% Y/% m/% d/', validators=[website.validators.validate_file_extension]),
        ),
    ]
