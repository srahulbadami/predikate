"""artilizer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import include, url
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('accounts/login/', views.user_login, name='user_login'),
    path('predictor/', views.predictor, name='predictor'),
    path('train/', views.train, name='train'),
    path('accounts/register/', views.register, name='register'),
    path('accounts/logout/', views.user_logout, name='user_logout'),
    path('fakenews_train/', views.fakenews_datacollect,name='fakenews_datacollect'),
    path('fakenews_predict/', views.fakenews_predict,name='fakenews_predict'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

