import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

# read the data
df=pd.read_csv('news.csv')

#  shape and head
print(df.shape)
print(df.head(5))

# the labels
labels = df.label
print(labels)

# splitting of data
x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# initilaize TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

#passive aggresive classifier

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])
