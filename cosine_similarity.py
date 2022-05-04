import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt') 
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwords = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report


tweets_data = pd.read_csv('/Users/nedamoussavi/Documents/CSC439SP22/final_project/Corona_tweets.csv', \
encoding = 'latin1')
tweets_data.shape
feat = tweets_data.text

tokenized_words = []
for i in range(0, len(feat)):
    tokenized_words.append(word_tokenize(feat[i]))
tokenized_words[1]

process_sents = []
for i in tokenized_words[1]:
    if i not in stopwords:
        process_sents.append(i)
preprocessing = []
for sentence in range(0, len(feat)):
    processed = re.sub(r'\@.*?\s','', str(feat[sentence])) 
    processed = re.sub(r'\W', ' ', str(processed)) 
    processed = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed) 
    processed = re.sub(r'\s+', ' ', processed, flags = re.I) 
    processed = re.sub(r'^b\s+', '', processed) 
    processed = processed.lower()
    preprocessing.append(processed)
docs = set(preprocessing) 
list(docs)[:3]

tfidf_vec = TfidfVectorizer()
tfidf_mat = tfidf_vec.fit_transform(docs)
#print(tfidf_mat.shape)

similar = cosine_similarity(tfidf_mat[6:7], tfidf_mat)
print(preprocessing[6]) 

similar = similar.reshape(-1)

n = 6 
for i in sorted(range(len(similar)), key = lambda sub: similar[sub])[-n:] :
    print(preprocessing[i])
    print('\n')