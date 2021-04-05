#trying to run the sentiment model first
# refer https://github.com/MarleyKuinn18/RecommendationSystem-v1/blob/main/Sentiment_Analysis/NaiveBayes.ipynb for details

import numpy as np
import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

dataset = pd.read_csv('Twitter_Data.csv')

comment = list(dataset['clean_text'])
sent = list(dataset['category'])
comment_train, comment_test, sent_train, sent_test = train_test_split(comment, sent, test_size=0.2)


tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = nltk.stem.RSLPStemmer()

def getCleanComment(review):
    review = str(review)
    review = review.lower()
    review = review.replace('"','')
    review = review.replace(';','')
    review = review.replace('-','')
    review = review.replace(',','')
    review = re.sub('\d', '', review)
    tokens = tokenizer.tokenize(review)
    new_tokens = [i for i in tokens if i not in en_stopwords]
    stemmed_tokens = [ps.stem(i) for i in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    return cleaned_review

comment_train = [getCleanComment(i) for i in comment_train]

cv = CountVectorizer()
comment_train_vec = cv.fit_transform(comment_train)
print(comment_train_vec.shape)
#print(cv.get_feature_names())

mnb = MultinomialNB()
mnb.fit(comment_train_vec,sent_train)

