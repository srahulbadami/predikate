# Generated by Django 2.2.7 on 2019-11-09 08:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0003_auto_20191108_2150'),
    ]

    operations = [
        migrations.CreateModel(
            name='CustomModels',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('upload', models.FileField(upload_to='data/')),
                ('model', models.CharField(blank=True, max_length=255, null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('custom_url', models.CharField(max_length=255)),
            ],
        ),
    ]
