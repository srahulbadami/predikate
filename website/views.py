from django.shortcuts import render

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
# Create your views here.
import string
from nltk.corpus import stopwords

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import _pickle as cPickle

@csrf_exempt
def fakenews_predict(request):
	if request.method == 'POST':
		text = [request.POST['data']]
		with open('website/models/model.pkl', 'rb') as fin:
			tfidf_vectorizer, pac = cPickle.load(fin)
			tfidf_test = tfidf_vectorizer.transform(text)
			y_pred=pac.predict(tfidf_test)
			print(y_pred)
		return HttpResponse(y_pred)

@csrf_exempt
def fakenews_train(request):
	if request.method == 'POST':
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
		return HttpResponse("Trained")