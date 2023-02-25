import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from PIL import Image
import snscrape.modules.twitter as sntwitter
import datetime
import pandas as pd
from datetime import datetime
import nltk
import string
from nltk.corpus import stopwords 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt



# This file contains modules that will be used in another python file preset in this project folder 
# (by importing)


def mapper( data):
    if data > 0:
        return "Positive"
    elif data < 0:
        return "Negative"
    else:
        return "Neutral"

def sentiment_analysis(df, col ):
    SIA = SentimentIntensityAnalyzer()
    df['polarity_scores'] = df[col].apply(lambda w: SIA.polarity_scores(w))
    df['compound_score'] = df['polarity_scores'].apply(lambda x: x['compound'])
    df['sentiment'] = df['compound_score'].apply(mapper)    
    return df




def preprocess_for_analysis(text):
    no_punc = [w.lower()  for w in text.split() if w.lower() not in string.punctuation]
    return  " ".join(no_punc)


def preprocess(text):
    no_punc = [w.lower()  for w in text.split() if w.lower() not in string.punctuation]
    no_stopwords = [w for w in no_punc  if w not in stopwords.words('english')]
    
    return " ".join(no_stopwords)

def get_cleaned_tweets(quote):

    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(quote).get_items():
        tweets.append([tweet.date, tweet.user.username , tweet.content])

    df = pd.DataFrame(tweets , columns= ['Date' , 'User' ,'tweet'])
    df['date_time'] = pd.to_datetime(df['Date'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['time'] = df['date_time'].dt.time

    del df['Date']
    del df['date_time']

    df['no_punc_tweets'] = df['tweet'].apply(preprocess_for_analysis) 
    #cleaned_text in new column
    df['cleaned_text_for_wordcloud'] = df['tweet'].apply(preprocess)
    df = sentiment_analysis(df , 'no_punc_tweets')
    return df 

def get_pos_neg_neu(df):
    pos = df[df['sentiment']=='Positive']
    neg = df[df['sentiment']=='Negative']
    neu = df[df['sentiment']=='Neutral']
    return pos , neg , neu


def get_hashtags(topic):
    hashtags = []
    topic = f'(#{topic}) lang:en'
    for tweet in sntwitter.TwitterSearchScraper(topic).get_items():
        if len(hashtags) ==200:
            break
        hashtags.append(tweet.hashtags)
    text = []
    for h in hashtags:
        if h is not None:
            text.append(' '.join([t for t in h]))
    return "".join(t for t in text)


def plot_wordcloud(text):
    wordcloud = WordCloud(max_words=5000, contour_width=3, contour_color='steelblue', background_color='white',min_font_size=10).generate(text)
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return fig