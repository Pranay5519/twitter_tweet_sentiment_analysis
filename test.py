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
import twitter



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
    
