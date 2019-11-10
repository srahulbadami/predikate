#!/usr/bin/env python
# coding: utf-8

# In[39]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import _pickle as cPickle


# In[ ]:





# In[40]:


def Pre_processing():
    # Importing the dataset
    dataset = pd.read_csv('affair.csv')
    X = dataset.iloc[:, :-1].values    
    
    # Taking care of missing data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    for i in range(len(X[0])):
        if str(type(X[0][i]))!= "<class 'str'>":
            imputer = imputer.fit(X[:, i:i+1])
            X[:, i:i+1] = imputer.transform(X[:, i:i+1])
    
    # Labelencoding and onehotencoding on categorical data
    a = list()
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    for i in range(len(X[0])):
        if str(type(X[0][i]))== "<class 'str'>":
            labelencoder_X = LabelEncoder()
            X[:, i] = labelencoder_X.fit_transform(X[:, i])
            a.append(i)
    onehotencoder = OneHotEncoder(categorical_features = a)
    X = onehotencoder.fit_transform(X).toarray()

    #Standarisation of data
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X = scaler.fit_transform(X)
    
    return(X)


# In[41]:


test = Pre_processing()


# In[42]:


X


# In[45]:


def Train_news():
    # Importing the dataset
    dataset = pd.read_csv('affair.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, len(dataset.columns)-1].values
    
    
    # Taking care of missing data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    for i in range(len(X[0])):
        if str(type(X[0][i]))!= "<class 'str'>":
            imputer = imputer.fit(X[:, i:i+1])
            X[:, i:i+1] = imputer.transform(X[:, i:i+1])
    
    # Labelencoding and onehotencoding on categorical data
    a = list()
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    for i in range(len(X[0])):
        if str(type(X[0][i]))== "<class 'str'>":
            labelencoder_X = LabelEncoder()
            X[:, i] = labelencoder_X.fit_transform(X[:, i])
            a.append(i)
    onehotencoder = OneHotEncoder(categorical_features = a)
    X = onehotencoder.fit_transform(X).toarray()

    
    #Standarisation of data
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X = scaler.fit_transform(X)
    
    #Dividing dataset into test and train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    #Using RandomForestClassifer model
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier = classifier.fit(X_train, y_train)
    
    score = classifier.score(X_test, y_test)
    print("Accuracy of Model is ","{0:.2f}".format(round(score*100,2)),"%")

    with open('model.pkl', 'wb') as fid:
        cPickle.dump((classifier,X_test,y_test), fid)


# In[ ]:





# In[46]:


Train_news()


# In[47]:


def predict_news(pred_data):
    with open('model.pkl', 'rb') as fin:
        classifier,X_test,y_test = cPickle.load(fin)
        
        #preprocessing the Pred_data
        #pred_data = Pre_processing()
        
        y_pred = classifier.predict(pred_data)
        print(y_pred)
        

    


# In[51]:


predict_news(test)


# In[ ]:


''' # dependent value Encoding    
    if str(type(y[0][0]))== "<class 'str'>":
        labelencoder_y = LabelEncoder()
        y = labelencoder_y.fit_transform(y)
    y = np.reshape(y, (-1, 1))
    onehotencoder = OneHotEncoder(categorical_features = [0])
    y = onehotencoder.fit_transform(y).toarray()
    
'''


# In[ ]:



def model_selection():
    
    #Fitting Linear Regression
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg = lin_reg.fit(X, y)
    
    
    # Fitting Polynomial Regression to the dataset
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)
    
    
    # Fitting Decision Tree Regression to the dataset
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X, y)
    
    
    # Fitting Random Forest Regression to the dataset
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X, y)

    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Fitting K-NN to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    
    
    # Fitting SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)


    # Fitting Naive Bayes to the Training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    
    # Fitting Decision Tree Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




