from django.shortcuts import render, redirect

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
# Create your views here.
import string
from nltk.corpus import stopwords

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import _pickle as cPickle
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import SignUpForm
from django.shortcuts import render_to_response
import requests
import bs4
import json
from .models import CustomModels
import os
def index(request):
	return render(request,'index.html')

@login_required
def train(request):
	if request.method == 'POST':
		for key in request.POST:
		    print(key)
		    value = request.POST[key]
		    print(value)
		model = CustomModels()
		data = request.FILES['myfile']
		name = request.POST['name']
		model.user = request.user
		model.name = name
		model.upload = data
		model.save()
		f=open(model.upload.url.strip("/"))
		f.seek(0)
		ext = data.name.split(".")[-1]
		i=1
		if ext=='csv':
			seperator=','
		elif ext=='tsv':
			seperatordata='	'
		data={}
		data_list = []
		for line in f:
			var = line.split(seperator)	
			data2 = {}
			j=1
			for a in var:
				data2['data_'+str(j)]= a
				j+=1
			data_list.append(data2)
			i=i+1
			if i==6:
				break
		return JsonResponse(json.dumps(data_list),safe=False)
	else:
		return render(request,'train.html')

def validate_file_extension(value):
    from django.core.exceptions import ValidationError
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.csv', '.tsv']
    if not ext.lower() in valid_extensions:
        raise ValidationError(u'Unsupported file extension.')

def user_login(request):
	print(request.path)
	if request.method == 'POST':
		username = request.POST.get('email')
		password = request.POST.get('password')
		user = authenticate(email=username, password=password)
		if user:
			if user.is_active:
				login(request,user)
				return HttpResponseRedirect(reverse('index'))
			else:
				return HttpResponse("Your account was inactive.")
		else:
			print("Someone tried to login and failed.")
			print("They used email: {} and password: {}".format(username,password))
			return HttpResponse("Invalid login details given")
	else:
		return render(request,'login.html')

@login_required
def user_logout(request):
	logout(request)
	return HttpResponseRedirect(reverse('index'))


@login_required
def predictor(request):
	if request.method == 'POST':
		if request.POST['type'] == 'fakenews':
			if request.POST['data'] == '':
				return JsonResponse("No Input",safe=False)
			res = fakenews_predict(request.POST['data'])
			return JsonResponse(res[0],safe=False)
	else:
		try:
			if request.GET['type']=='fakenews':
				return render(request,'predict.html', {'name': 'Fake News','type':'fakenews','data':''})
			return HttpResponseRedirect(reverse('index'))
		except:
			return HttpResponseRedirect(reverse('index'))

def fakenews_predict(var):
	text = [var]
	with open('website/models/model.pkl', 'rb') as fin:
		tfidf_vectorizer, pac = cPickle.load(fin)
		tfidf_test = tfidf_vectorizer.transform(text)
		y_pred=pac.predict(tfidf_test)
		print(y_pred)
	return y_pred

@csrf_exempt
def fakenews_datacollect(request):
	if request.method == 'POST':
		f=open("website/input/news.csv", "a+")
		f.write("text,label\n")
		for i in range(1,443):
			url = 'https://www.jagranjosh.com/current-affairs/monthly-1483099181-catlistshow-1-p' + str(i)
			response = requests.get(url)
			soup = bs4.BeautifulSoup(response.text,features="html.parser")
			data = soup.select('div.articlelanding_detail a')
			data = [a.attrs.get('title') for a in data]
			for info in data:
				var = info.replace("'","" )
				var = var.replace(",",";")
				f.write(var+","+"REAL\n")
			for i in range(1,196):
				url = 'https://www.jagranjosh.com/current-affairs/national-india-1283851987-catlistshow-1-p' + str(i)
				response = requests.get(url)
				soup = bs4.BeautifulSoup(response.text,features="html.parser")
				data = soup.select('div.articlelanding_detail a')
				data = [a.attrs.get('title') for a in data]
			for info in data:
				var = info.replace("'","" )
				var = var.replace(",",";")
				f.write(var+","+"REAL\n")
			for i in range(1,194):
				url = 'https://www.jagranjosh.com/current-affairs/international-world-1283850903-catlistshow-1-p' + str(i)
				response = requests.get(url)
				soup = bs4.BeautifulSoup(response.text,features="html.parser")
				data = soup.select('div.articlelanding_detail a')
				data = [a.attrs.get('title') for a in data]
			for info in data:
				var = info.replace("'","" )
				var = var.replace(",",";")
				f.write(var+","+"REAL\n")
			for i in range(1,148):
				url = 'https://www.jagranjosh.com/current-affairs/sports-1286371683-catlistshow-1-p' + str(i)
				response = requests.get(url)
				soup = bs4.BeautifulSoup(response.text,features="html.parser")
				data = soup.select('div.articlelanding_detail a')
				data = [a.attrs.get('title') for a in data]
			for info in data:
				var = info.replace("'","" )
				var = var.replace(",",";")
				f.write(var+","+"REAL\n")
			for i in range(1,71):
				url = 'https://www.jagranjosh.com/current-affairs/science-and-technology-1286444078-catlistshow-1-p' + str(i)
				response = requests.get(url)
				soup = bs4.BeautifulSoup(response.text,features="html.parser")
				data = soup.select('div.articlelanding_detail a')
				data = [a.attrs.get('title') for a in data]
			for info in data:
				var = info.replace("'","" )
				var = var.replace(",",";")
				f.write(var+","+"REAL\n")
			for i in range(1,145):
				url = 'https://www.jagranjosh.com/current-affairs/economy-1284037727-catlistshow-1-p' + str(i)
				response = requests.get(url)
				soup = bs4.BeautifulSoup(response.text,features="html.parser")
				data = soup.select('div.articlelanding_detail a')
				data = [a.attrs.get('title') for a in data]
			for info in data:
				var = info.replace("'","" )
				var = var.replace(",",";")
				f.write(var+","+"REAL\n")
			for i in range(1,64):
				url = 'https://www.jagranjosh.com/current-affairs/environment-and-ecology-1286444215-catlistshow-1-p' + str(i)
				response = requests.get(url)
				soup = bs4.BeautifulSoup(response.text,features="html.parser")
				data = soup.select('div.articlelanding_detail a')
				data = [a.attrs.get('title') for a in data]
			for info in data:
				var = info.replace("'","" )
				var = var.replace(",",";")
				f.write(var+","+"REAL\n")
		f.close()
		if(fakenews_train()==0):
			return HttpResponse("Trained with new News")
		else:
			return HttpResponse("Failed")


def register(request):
	if request.user.is_authenticated:
		return redirect('index')
	if request.method == 'POST':
		form = SignUpForm(request.POST)
		if form.is_valid():
			user = form.save()
			raw_password = form.cleaned_data.get('password1')
			user = authenticate(email=user.email, password=raw_password)
			login(request, user)
			return redirect('index')
	else:
		form = SignUpForm()
	return render(request, 'registration.html', {'form': form})


def fakenews_train():
	# try:
	data=pd.read_csv('website/input/news.csv')
	labels=data.label
	X = data['text']
	#processing the data
	tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
	tfidf_train = tfidf_vectorizer.fit_transform(X)
	
	pac=PassiveAggressiveClassifier(max_iter=50)
	pac = pac.fit(tfidf_train,labels)
	
	with open('website/models/model.pkl', 'wb') as fid:
		cPickle.dump((tfidf_vectorizer,pac), fid)
	return(0)
	# except:
	# 	return(-1)