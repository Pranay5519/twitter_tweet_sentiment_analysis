"""
Twitter uses various types of user data to recommend tweets to its users. These include:

User engagement: Twitter tracks the tweets, accounts, and topics that a user has engaged with in the past, such as likes, retweets, and replies, to understand their interests and preferences.

User demographics: Twitter may also use demographic data such as the user's location, age, gender, and language to recommend tweets that are more relevant to them.

User connections: Twitter takes into account a user's connections on the platform, such as the accounts they follow and the users who follow them, to recommend tweets that are popular among their network.

User activity: Twitter may also use data on a user's activity, such as the time of day they are most active, to recommend tweets that are more likely to be seen and engaged with.

User search history: Twitter may use a user's search history to recommend tweets that are relevant to their recent searches.

User device: Twitter may also take into account the device that a user is using, such as a smartphone or a desktop computer, to recommend tweets that are optimized for that device.

"""

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
import streamlit as st

class twitter:
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



st.title("Twitter Tweets Sentiment Analysis and WordCloud")
image = Image.open("twitterlogo.jpg")
st.sidebar.image(image)

topic = st.sidebar.text_input("",'hashtag topic')
hashtags = twitter.get_hashtags(topic)
wc__fig  = twitter.plot_wordcloud(hashtags)
if st.sidebar.button('Get hashtags'):
    st.pyplot(wc__fig)
    st.write(hashtags)
    

username = st.sidebar.text_input("Enter Twitter Username :" , '')
since_date= st.sidebar.date_input("Tweets since  (date):")
to_date = st.sidebar.date_input("tweets Until (date)  : " )
quote = f'(from:{username}) lang:en until:{to_date} since:{since_date}'
df = twitter.get_cleaned_tweets(quote)

display_df = df.drop(['no_punc_tweets',	'cleaned_text_for_wordcloud',	'polarity_scores'] , axis=1)
pos , neg , neu = twitter.get_pos_neg_neu(df)
if st.sidebar.button("scrape and show DataFrame"):
    st.subheader("Sentimental analysis")
    st.dataframe(display_df)
    
if st.sidebar.button('scrape and show wordcloud'):
    st.subheader('wordcloud')
    text = ' '.join(t for t in df['cleaned_text_for_wordcloud'])
    wordcloud = WordCloud( background_color='white',min_font_size=10).generate(text)
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)
    
if st.sidebar.button("Scrape and PIE PLOT"):
    st.subheader('PIEPLOT')
    fig , ax = plt.subplots()
    temp = pos['sentiment'].count() , neg['sentiment'].count() , neu['sentiment'].count()
    plt.pie(temp , labels = ['positive' , 'negative','neutral'], autopct = '%.2f' , shadow = True, explode =(0.02,0.02, 0.1))
    st.pyplot(fig)
    
