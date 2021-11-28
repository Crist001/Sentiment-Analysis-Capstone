
from nltk import text
import pandas as pd
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns
import re
import ssl
import nltk
from nltk.corpus.reader.knbc import test
from nltk.tokenize.regexp import WhitespaceTokenizer
ssl._create_default_https_context = ssl._create_stdlib_context
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus.reader import CorpusReader 
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import neattext.functions as nfx
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from collections import defaultdict
test_tweets = pd.read_csv('./testing_tweets_1.csv', delimiter=',')
train_tweets = pd.read_csv('./training_tweets_1.csv', delimiter=',', encoding= 'unicode_escape')

#print(train_tweets)
#make all lowercase
test_tweets["Cleaned_Tweet"] = test_tweets["Tweet"].str.lower()
train_tweets["Cleaned_Tweet"] = train_tweets["Tweet"].str.lower()
#print(test_tweets.head())
#print(train_tweets.head()) 


#remove numbers and special characters
p = re.compile (r"\d+")
test_tweets["Cleaned_Tweet"] = [p.sub ('NUM', x) for x in test_tweets["Cleaned_Tweet"]]
p2=re.compile("@[A-Za-z0-9_]+")
test_tweets["Cleaned_Tweet"] = [p2.sub ('', x) for x in test_tweets["Cleaned_Tweet"]]
test_tweets["Cleaned_Tweet"] = test_tweets["Cleaned_Tweet"].apply(nfx.remove_emojis)
test_tweets["Cleaned_Tweet"] = test_tweets["Cleaned_Tweet"].apply(nfx.remove_multiple_spaces)
test_tweets["Cleaned_Tweet"] = test_tweets["Cleaned_Tweet"].apply(nfx.remove_urls)
test_tweets["Cleaned_Tweet"] = test_tweets["Cleaned_Tweet"].apply(nfx.remove_punctuations)
test_tweets["Cleaned_Tweet"] = test_tweets["Cleaned_Tweet"].apply(nfx.remove_special_characters)
p3=re.compile("https*")
test_tweets["Cleaned_Tweet"] = [p3.sub ('', x) for x in test_tweets["Cleaned_Tweet"]]

p4 = re.compile (r"\d+")
train_tweets["Cleaned_Tweet"] = [p4.sub ('NUM', x) for x in train_tweets["Cleaned_Tweet"]]
p5=re.compile("@[A-Za-z0-9_]+")
train_tweets["Cleaned_Tweet"] = [p5.sub ('', x) for x in train_tweets["Cleaned_Tweet"]]
train_tweets["Cleaned_Tweet"] = train_tweets["Cleaned_Tweet"].apply(nfx.remove_emojis)
train_tweets["Cleaned_Tweet"] = train_tweets["Cleaned_Tweet"].apply(nfx.remove_multiple_spaces)
train_tweets["Cleaned_Tweet"] = train_tweets["Cleaned_Tweet"].apply(nfx.remove_urls)
train_tweets["Cleaned_Tweet"] = train_tweets["Cleaned_Tweet"].apply(nfx.remove_punctuations)
train_tweets["Cleaned_Tweet"] = train_tweets["Cleaned_Tweet"].apply(nfx.remove_special_characters)
p6=re.compile("https*")
train_tweets["Cleaned_Tweet"] = [p6.sub ('', x) for x in train_tweets["Cleaned_Tweet"]]
#print(test_tweets.head())
#print(train_tweets.head())

#remove stop words
stop_words=stopwords.words('english')
test_tweets['Cleaned_Tweet'] = test_tweets['Cleaned_Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
train_tweets['Cleaned_Tweet'] = train_tweets['Cleaned_Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#print(test_tweets.head())
#print(train_tweets.head())

#Tokenization
vectorizer = CountVectorizer()
cleaned__test_tweet= test_tweets["Cleaned_Tweet"]
vectorized_test_tweets = vectorizer.fit_transform(cleaned__test_tweet)
test_tweets_array = vectorized_test_tweets.toarray()
#print(vectorizer.get_feature_names_out())
#print(vectorized_test_tweets.toarray())
#print(vectorized_test_tweets.shape)
cleaned_test_tweets=test_tweets["Cleaned_Tweet"]  


cleaned_train_tweet= train_tweets["Cleaned_Tweet"]
vectorized_train_tweets = vectorizer.fit_transform(cleaned_train_tweet)
#print(vectorizer.get_feature_names_out())
#print(vectorized_test_tweets.toarray())
#print(vectorized_test_tweets.shape)
cleaned_train_tweets=train_tweets["Cleaned_Tweet"]


#TFIDF
tfidf=TfidfVectorizer()
tfidf_train_tweets=tfidf.fit_transform(cleaned_train_tweet)
print(type(tfidf_train_tweets))
print(len(tfidf.vocabulary_))
print(tfidf.vocabulary_)
print(tfidf_train_tweets)
tfidf_train_tweets_array = tfidf_train_tweets.toarray()
print(type(tfidf_train_tweets_array))
print(tfidf_train_tweets_array)


tfidf_test_tweets=tfidf.transform(cleaned__test_tweet)
print(type(tfidf_test_tweets))
print(len(tfidf.vocabulary_))
print(tfidf.vocabulary_)
print(tfidf_test_tweets)
tfidf_test_tweets_array = tfidf_test_tweets.toarray()
print(type(tfidf_test_tweets_array))
print(tfidf_test_tweets_array.shape)


# Train Model
#assign training and testing
x_test=tfidf_test_tweets_array
x_train=tfidf_train_tweets_array
y_train=train_tweets['Label']
model=LogisticRegression(C=1,penalty='l1', solver='saga').fit(x_train, y_train)
print(model.predict_proba(x_test))
y_predict = [int(p[1]>0.5)for p in model.predict_proba(x_test)]
print(y_predict)


y_predict_train = [int(p[1]>0.5)for p in model.predict_proba(x_train)]
print(accuracy_score(y_train, y_predict_train))

print(len(model.coef_[0]))
feature_names = tfidf.get_feature_names()
for i,col in enumerate (feature_names):
  if np.abs(model.coef_[0, i])>1e9:  
    print(feature_names[i], ' - ', model.coef_[0,i])
print(len(feature_names))



#Classification Report
print(classification_report(y_train, y_predict_train))
print(confusion_matrix(y_train, y_predict_train))



















