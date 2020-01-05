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
from datetime import datetime
from sendfile import sendfile
from wsgiref.util import FileWrapper
import mimetypes
from background_task import background

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

def index(request):
	return render(request,'index.html')

def logs(request):
	if request.user.is_authenticated:
		if request.user.is_admin:
			return render(request,'logs.html')
	return render(request,'index.html')

def error_404(request,exception):
	data = {}
	return render(request,'error_404.html', data)

def error_500(request):
	data = {}
	return render(request,'error_500.html', data)

@login_required
def download(request):
	if request.method=='GET':
		try:
			user=request.user
			modelId = request.GET.getlist('download_query')
			model = CustomModels.objects.get(id=modelId[0])
			if model.user == user:
				return sendfile(request, model.upload.url.strip('/'), attachment=True, attachment_filename='dataset.'+model.upload.name.split(".")[-1])
			else:
				user=request.user
				models = CustomModels.objects.filter(user=user)
				return render(request,'models.html',{'data': models})
		except:
			user=request.user
			models = CustomModels.objects.filter(user=user)
			return render(request,'models.html',{'data': models})
	

# send myfile.pdf to user


@login_required
def custommodels(request):
	if request.method == 'POST':
		print(request.FILES)
		data = request.FILES['myfile']
		modelid = request.POST['modelId']
		model = CustomModels.objects.get(id=modelid)
		custom = model.cus_model.url.strip("/")
		model.temp_data = data
		model.save()
		pre = Pre_processing(model.temp_data.url.strip("/"))
		predicteddata = predict(custom,pre)
		data = pd.read_csv(model.temp_data.url.strip("/"))
		data['result'] = predicteddata
		data.to_csv(model.temp_data.url.strip("/"))
		return sendfile(request, model.temp_data.url.strip('/'), attachment=True, attachment_filename='dataset.'+model.upload.name.split(".")[-1])
	else:
		user=request.user
		models = CustomModels.objects.filter(user=user)
		return render(request,'models.html',{'data': models})
@login_required
def train(request):
	if request.method == 'POST':
		model = CustomModels()
		data = request.FILES['myfile']
		name = request.POST['name']
		type_model = request.POST['type']
		a = ['Linear Regression','Polynomial Features','Decision Tree Regressor','Logistic Regression','KNeighbors Classifier','SVC','GaussianNB','DecisionTreeClassifier','RandomForestClassifier','SVC_2','Random Forest Regressor']	
		model.user = request.user
		model.name = name
		model.upload = data
		model.model_used = type_model
		model.model_used_name = a[int(type_model)-1]
		model.save()
		train_custom_model(model.id,type_model)
		data = model.upload
		f=open(model.upload.url.strip("/"))
		f.seek(0)
		ext = data.name.split(".")[-1]
		i=1
		if ext=='csv':
			seperator=','
		elif ext=='tsv':
			seperatordata='	'
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
		data_list = []
		data2 = {}
		data2['model'] = 'Linear Regression'
		data2['id'] = 1
		data_list.append(data2)
		# data2 = {}
		# data2['model'] = 'Polynomial Features'
		# data2['id'] = 2
		# data_list.append(data2)
		data2 = {}
		data2['model'] = 'Decision Tree Regressor'
		data2['id'] = 3
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'Logistic Regression'
		data2['id'] = 4
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'KNeighbors Classifier'
		data2['id'] = 5
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'SVC'
		data2['id'] = 6
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'GaussianNB'
		data2['id'] = 7
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'Decision Tree Classifier'
		data2['id'] = 8
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'Random Forest Classifier'
		data2['id'] = 9
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'SVC_2'
		data2['id'] = 10
		data_list.append(data2)
		data2 = {}
		data2['model'] = 'Random Forest Regressor'
		data2['id'] = 11
		data_list.append(data2)
		return render(request,'train.html',{'data':data_list})


@background(schedule=5)
def train_custom_model(id_model,type_model):
	model = CustomModels.objects.get(id=id_model)
	channel_layer = get_channel_layer()
	print(datetime.now())
	async_to_sync(channel_layer.group_send)(
		"gossip", {"type": "user.gossip",
				   "event": "Started",
				   "now": str(datetime.now()),
				   "username": model.user.first_name + " " +model.user.last_name})
	print("WORKING ON IT")
	# test = Pre_processing(model.upload.url.strip("/"))
	# location= 'media/premodels/'+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.pkl'
	# with open(location, 'wb') as fid:
	# 	cPickle.dump(test, fid)
	location= 'premodels/'+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.pkl'
	model.pre_model = location
	location2,acc = training(model.upload.url.strip("/"),type_model)
	model.cus_model = location2
	model.accuracy = acc
	model.save()
	print("WORKING Complete")
	channel_layer = get_channel_layer()
	async_to_sync(channel_layer.group_send)(
		"gossip", {"type": "user.gossip",
				   "event": "Ended",
				   "now": str(datetime.now()),
				   "username": model.user.first_name + " " +model.user.last_name})
	return("Done")

def validate_file_extension(value):
	from django.core.exceptions import ValidationError
	ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
	valid_extensions = ['.csv', '.tsv']
	if not ext.lower() in valid_extensions:
		raise ValidationError(u'Unsupported file extension.')

def user_login(request):
	if request.user.is_authenticated:
		return redirect('index')
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

def Pre_processing(data):
	# Importing the dataset
	dataset = pd.read_csv(data)
	X = dataset.iloc[:, :].values    
	
	# Taking care of missing data
	from sklearn.impute import SimpleImputer
	# from sklearn.preprocessing import Imputer

	# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
	imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
	for i in range(len(X[0])):
		if str(type(X[0][i]))!= "<class 'str'>":
			imputer = imputer.fit(X[:, i:i+1])
			X[:, i:i+1] = imputer.transform(X[:, i:i+1])
	
	# Labelencoding and onehotencoding on categorical data
	a = list()
	flag = 0
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	from sklearn.compose import ColumnTransformer
	for i in range(len(X[0])):
		if str(type(X[0][i]))== "<class 'str'>":
			flag = 1
			# labelencoder_X = LabelEncoder()
			# X[:, i] = labelencoder_X.fit_transform(X[:, i])
			a.append(i)
	if flag == 1 :
		ct = ColumnTransformer(
			[('one_hot_encoder', OneHotEncoder(), a)],remainder='passthrough')
		X = ct.fit_transform(X)

	# #Standarisation of data
	# from sklearn.preprocessing import StandardScaler
	# scaler=StandardScaler()
	# X = scaler.fit_transform(X)
	
	return(X)




def training(location1,type_model):
	# Importing the dataset
	dataset = pd.read_csv(location1)
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, len(dataset.columns)-1].values
	
	
	# Taking care of missing data
	# from sklearn.preprocessing import Imputer

	# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
	from sklearn.impute import SimpleImputer
	imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
	for i in range(len(X[0])):
		if str(type(X[0][i]))!= "<class 'str'>":
			imputer = imputer.fit(X[:, i:i+1])
			X[:, i:i+1] = imputer.transform(X[:, i:i+1])
	
	# Labelencoding and onehotencoding on categorical data
	a = list()
	flag = 0
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	from sklearn.compose import ColumnTransformer
	for i in range(len(X[0])):
		if str(type(X[0][i]))== "<class 'str'>":
			flag = 1
			# labelencoder_X = LabelEncoder()
			# X[:, i] = labelencoder_X.fit_transform(X[:, i])
			a.append(i)
	if flag == 1:
		# onehotencoder = OneHotEncoder(categorical_features = a)
		# X = onehotencoder.fit_transform(X).toarray()
		ct = ColumnTransformer(
			[('one_hot_encoder', OneHotEncoder(), a)],remainder='passthrough')
		X = ct.fit_transform(X)
	# if str(type(y[0][0]))=="<class 'str'>":
	# 	labelencoder_y = LabelEncoder()
	# 	y = labelencoder_y.fit_transform(y)

	
	# #Standarisation of data
	# from sklearn.preprocessing import StandardScaler
	# scaler=StandardScaler()
	# X = scaler.fit_transform(X)
	
	#Dividing dataset into test and train
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	
	#Using RandomForestClassifer model
	# from sklearn.ensemble import RandomForestClassifier
	# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
	# classifier = classifier.fit(X_train, y_train)

	# Fitting Logistic Regression to the Training set
	type_model = int(type_model)
	if type_model==1:
		from sklearn.linear_model import LinearRegression
		classifier = LinearRegression()
		classifier = classifier.fit(X, y)
	# elif type_model==2:
	# 	from sklearn.preprocessing import PolynomialFeatures
	# 	poly_reg = PolynomialFeatures(degree = 4)
	# 	X_poly = poly_reg.fit_transform(X)
	# 	poly_reg.fit(X_poly, y)
	# 	classifier = LinearRegression()
	# 	classifier.fit(X_poly, y)
	elif type_model==3:
		from sklearn.tree import DecisionTreeRegressor
		classifier = DecisionTreeRegressor(random_state = 0)
		classifier.fit(X, y)
	elif type_model==4:
		from sklearn.linear_model import LogisticRegression
		classifier = LogisticRegression(random_state = 0)
		classifier = classifier.fit(X_train, y_train)
	elif type_model==5:
		from sklearn.neighbors import KNeighborsClassifier
		classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
		classifier.fit(X_train, y_train)
	elif type_model==6:
		from sklearn.svm import SVC
		classifier = SVC(kernel ='linear', C = 1).fit(X_train, y_train) 
		print("Here")
	elif type_model==7:
		from sklearn.naive_bayes import GaussianNB
		classifier = GaussianNB()
		classifier.fit(X_train, y_train)
	elif type_model==8:
		from sklearn.tree import DecisionTreeClassifier
		classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
		classifier.fit(X_train, y_train)
	elif type_model==9:
		from sklearn.ensemble import RandomForestClassifier
		classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
		classifier.fit(X_train, y_train)

	elif type_model==10:
		from sklearn.svm import SVC
		classifier = SVC(kernel = 'rbf', random_state = 0)
		classifier.fit(X_train, y_train)
		print("HERE")
	elif type_model==11:
		from sklearn.ensemble import RandomForestRegressor
		classifier = RandomForestRegressor(n_estimators = 10, random_state = 0)
		classifier.fit(X, y)
	score = classifier.score(X_test, y_test)
	acc = round(score*100,2)
	location= 'data/model/'+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.pkl'
	with open('media/' + location, 'wb') as fid:
		cPickle.dump((classifier,X_test,y_test), fid)
	return(location,acc)

def predict(data,pred_data):
	with open(data, 'rb') as fin: #Trained Data
		classifier,X_test,y_test = cPickle.load(fin)
		y_pred = classifier.predict(pred_data)
		print(y_pred)
		return(y_pred)
		
