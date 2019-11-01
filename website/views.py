from django.shortcuts import render

import pandas as pd
import _pickle as cPickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
# Create your views here.
import string
from nltk.corpus import stopwords

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,HttpResponse

def text_process(mess):
	"""
	Takes in a string of text, then performs the following:
	1. Remove all punctuation
	2. Remove all stopwords
	3. Returns a list of the cleaned text
	"""
	# Check characters to see if they are in punctuation
	nopunc = [char for char in mess if char not in string.punctuation]

	# Join the characters again to form the string.
	nopunc = ''.join(nopunc)
	
	# Now just remove any stopwords
	return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
@csrf_exempt
def predict(request):
	if request.method == 'POST':
		messages1 = request.POST['parag']
		with open('website/models/emotion_detect_model.pkl', 'rb') as fin:
		  bow_transformer, emotion_detect_model = cPickle.load(fin)

		# fid = open('', 'rb')
		# emotion_detect_model = cPickle.load(fid)


		messages_bow_test = bow_transformer.transform((messages1.split('.')))


		tfidf_transformer = TfidfTransformer()

		messages_tfidf_test = tfidf_transformer.fit_transform(messages_bow_test)

		all_predictions = emotion_detect_model.predict(messages_tfidf_test)
		print(all_predictions)
		return HttpResponse(all_predictions)


def train(request):
	print("Working")
	messages = pd.read_csv('website/input/data.bak3.tsv', sep='\t',
						   names=["label", "message"])
	bow_transformer = CountVectorizer(analyzer=text_process)
	X_train = bow_transformer.fit(messages['message'])
	messages_bow = X_train.transform(messages['message'])
	tfidf_transformer = TfidfTransformer()
	tf = tfidf_transformer.fit(messages_bow)
	messages_tfidf = tf.transform(messages_bow)
	emotion_detect_model = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None).fit(messages_tfidf, messages['label'])
	with open('website/models/emotion_detect_model.pkl', 'wb') as fid:
		cPickle.dump((bow_transformer,emotion_detect_model), fid) 
	print("Done")