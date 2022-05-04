import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
analyzer = SentimentIntensityAnalyzer()


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report,confusion_matrix 
import matplotlib.pyplot as pltcm 


tweets_data = pd.read_csv('/Users/nedamoussavi/Documents/CSC439SP22/final_project/Corona_tweets.csv', \
    encoding = 'latin1')
tweets_data.drop_duplicates()

def analyzer_score_sentiment(tweet):
    analyzer_score = analyzer.polarity_scores(tweet)
    print("{:-<40} {}".format(tweet, str(analyzer_score)))

tokenize_words = RegexpTokenizer(r'\w+')
words = tweets_data['text'].apply(tokenize_words.tokenize)
words.head()
compile_words = [w for t in words for w in t]
tweets_data['lengths'] = [len(t) for t in words]

tweets_data['analyzer_score'] = tweets_data['text'].apply(lambda twitter: analyzer.polarity_scores(twitter))
tweets_data['compound'] = tweets_data['analyzer_score'].apply(lambda score: score['compound'])

def sentiment_analysis(x):
    if x <= -0.05:
        return "Negative"
    elif x >= 0.05:
        return "Positive"
    else:
        return "Neutral"

tweets_data['sentiment analysis'] = tweets_data['compound'].apply(sentiment_analysis)
tweets_data = tweets_data[['text', 'sentiment analysis']]
en = LabelEncoder()
tweets_data['text'] = en.fit_transform(tweets_data['text'])


sent = tweets_data.groupby('sentiment analysis').count()['text'].reset_index().sort_values(by = 'text', ascending = False)
sent.style.background_gradient()
plt.figure(figsize = (12, 6))
sns.countplot(x = 'sentiment analysis', data = tweets_data, palette = 'inferno')
plt.show()


# logistic regression
SENT = tweets_data.drop(['sentiment analysis'], 1)
target = tweets_data['sentiment analysis']
#train 75  test 25 
SENT_train, SENT_test, target_train, target_test = train_test_split(SENT, target, \
test_size = 0.25, random_state = 45)

log_reg = LogisticRegression()
log_reg.fit(SENT_train, target_train)
predict = log_reg.predict(SENT_test)
print('Accuracy =', accuracy_score(target_test, predict))


# confusion matrix

confusion_matrix(target_test, predict)
print(classification_report(target_test, predict))
#conf_mat = confusion_matrix(target_test, predict)
plt.figure(figsize = (7, 7))
sns.heatmap(confusion_matrix(target_test, predict), 
            cmap = "YlGnBu", 
            linecolor = 'black', 
            linewidth = 1, 
            annot = True, 
            fmt = '', 
            xticklabels = ['neg','neut','pos'], 
            yticklabels = ['neg','neut','pos'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.plot()
plt.show()
