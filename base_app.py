"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
#General imports
import numpy as np
import pandas as pd
import itertools
import re
from advertools.emoji import extract_emoji

#Display for analytics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

# Streamlit dependencies
import streamlit as st
import joblib,os,base64
import seaborn as sns

# Data dependencies
import pandas as pd
import numpy as np
from nltk import FreqDist
from collections import Counter

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

#load in local css styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#extracting url,hastags and mentions
def url_hash_mention_extractor(df):
	"""
    This function extracts url, hashtag and user mentions
    from tweets.

    Arguments:
    dataframe containing tweets in 'message' column.

    Returns:
    a modified dataframe with url, hashtag and mention columns.
    """
	# initialise lists & strings
	urls = []
	tags = []
	ment = []
	m_str = '@(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	u = 'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	t_str = '#(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
	# extract & append information
	for tweet in df['message'].str.lower():
		ment.append(re.findall(m_str, tweet))
		urls.append(re.findall(u, tweet))
		tags.append(re.findall(t_str, tweet))
	# add extracted information to df
	df['urls'] = urls
	df['hashtags'] = tags
	df['mentions'] = ment
	return df


def common_hashtags(df, n):
	"""
    Returns a dataframe with the most common 'n' hashtags
    in tweet data and the frequency of said hashtags.

    Arguments:
    df: dataframe containing 'hashtag' column of extracted
    hashtags.

    n: integer number of common words

    Returns:
    dataframe of common hashtags and frequency of use.

    """
	hashtag_list = []
	for i in range(0, len(df)):  # extract hashtags
		hashtags = df['hashtags'][i]
		hashtags = [word for word in hashtags]
		hashtags = ''.join(hashtags)
		hashtag_list.append(hashtags)
	top_hashtags = Counter(hashtag_list).most_common(n + 1)
	comm_hashtag = pd.DataFrame(np.array(top_hashtags),  # create new df
								columns=['hashtag', 'frequency'])
	comm_hashtag['frequency'] = comm_hashtag['frequency'].astype(int)
	return comm_hashtag[1:]


def common_mentions(df, n):
	"""
    Returns a dataframe with the most common 'n' user mentions
    in tweet data and the frequency of said user mentions.

    Arguments:
    df: dataframe containing 'mentions' column of username
    mentions.

    n: integer number of common username mentions

    Returns:
    dataframe of common username mentions and frequency of use.

    """
	mentions_list = []
	for i in range(0, len(df)):  # extract mentions
		mentions = df['mentions'][i]
		mentions = [word for word in mentions]
		mentions = ''.join(mentions)
		mentions_list.append(mentions)
	top_mentions = Counter(mentions_list).most_common(n + 1)
	common_mentions = pd.DataFrame(np.array(top_mentions),  # create new df
								   columns=['mention', 'frequency'])
	common_mentions['frequency'] = common_mentions['frequency'].astype(int)
	return common_mentions[1:]

#extracting tweet emoji
def tweet_get_emojis(df):
    emoji_2d_list = df['message'].values
    emoji_2d_list = extract_emoji(emoji_2d_list)
    df['emoji'] = emoji_2d_list['emoji']
    return df

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages

	#loading data
	# Load your raw data
	raw = pd.read_csv("resources/train.csv")

	# extracting urls,hastags and mentions
	raw = url_hash_mention_extractor(raw)

	# extracting emoji
	raw = tweet_get_emojis(raw)

	#loading custom css
	local_css('styles/style.css')

	#setting up columns for length insights
	raw['no_of_characters'] = raw['message'].apply(len)
	raw['word_count'] = raw.message.str.split().apply(len)
	raw['av_word_length'] = raw['no_of_characters'] / raw['word_count']

	#setting up the sentiment classes for word freq
	anti = raw[(raw['sentiment']) == -1].reset_index(drop=True)
	neutral = raw[(raw['sentiment']) == 0].reset_index(drop=True)
	pro = raw[(raw['sentiment']) == 1].reset_index(drop=True)
	news = raw[(raw['sentiment']) == 2].reset_index(drop=True)

	#Title
	st.title("Tweet Classifier")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home Page","Prediction", "Data and Insights"]
	selection = st.sidebar.selectbox("Choose Option", options)
	# Building out the "Information" page
	if selection == "Data and Insights":
		st.info("Select insights")
		# You can read a markdown file from supporting resources folder
		insights = ['Raw Data','Data Balance','Tweet Length','Word Frequency','Popular Hastags','Popular Usernames','Popular Emojis']
		selection_info = st.selectbox("Select page", insights)

		if selection_info == "Raw Data":
			st.write(raw[['sentiment', 'message']])

		elif selection_info == "Data Balance":
			st.write('Data Balance: What is the distribution of tweets by sentiment class?')

			sns.set(style="ticks", context="talk")
			plt.style.use("dark_background")
			sns.set(font_scale=1.3)
			plt.figure(figsize=(10, 5))
			count = sns.countplot(x=raw['sentiment'], data=raw, palette="pastel")
			plt.title('Tweet distribution by Sentiment Class: Train data')

			total = len(raw)
			for p in count.patches:
				height = p.get_height()
				count.text(p.get_x() + p.get_width() / 2.,
						   height + 3,
						   '{:.0%}'.format(height / total),
						   ha="center")
			st.pyplot()

		if selection_info == "Tweet Length":  # data is hidden if box is unchecked
			sns.set(style="ticks", context="talk")
			plt.style.use("dark_background")
			plt.figure(figsize=(10, 6))
			sns.boxplot(x=raw['sentiment'],
						y=raw['no_of_characters'],
						data=raw,
						palette="pastel")
			plt.title('Characters per Tweet by Sentiment Class')
			plt.xlabel('Sentiment Class')
			plt.ylabel('Tweet Length (characters)');

			st.pyplot()

		if selection_info == "Word Frequency":  # data is hidden if box is unchecked

			#world cloud on most common tweets
			train_text = " ".join(tweet for tweet in raw.message)
			train_wordcloud = WordCloud(max_font_size=300,
										background_color="black",
										width=1600,
										height=800,
										collocations=False,
										colormap='Paired').generate(train_text)
			plt.figure(figsize=(20, 10))
			plt.title('The most common words in the tweet data', fontsize=40)
			plt.imshow(train_wordcloud)
			plt.axis("off")
			plt.tight_layout(pad=0)
			plt.show()
			st.pyplot()

			# determine most common words
			most_common_anti = list(itertools.chain
									(*[tweet.split() for tweet in
									   anti['message'].values]))
			freq = FreqDist(most_common_anti)
			y_label = [x[0] for x in freq.most_common(n=20)]
			x_freq = [x[1] for x in freq.most_common(n=20)]
			freq.most_common(n=20)

			# create plot of most common words for anti sentiment
			plt.figure(figsize=(15, 8))
			sns.barplot(data={'x': x_freq, 'y': y_label}, x='x', y='y')
			plt.xlabel('Frequency')
			plt.ylabel('Word')
			plt.title('Top 20 Words: Anti Sentiment',fontsize=20);

			#neutral
			most_common_neutral = list(itertools.chain
									(*[tweet.split() for tweet in
									   neutral['message'].values]))
			freq = FreqDist(most_common_neutral)
			y_label = [x[0] for x in freq.most_common(n=20)]
			x_freq = [x[1] for x in freq.most_common(n=20)]
			freq.most_common(n=20)

			# create plot of most common words for anti sentiment
			plt.figure(figsize=(15, 8))
			sns.barplot(data={'x': x_freq, 'y': y_label}, x='x', y='y')
			plt.xlabel('Frequency')
			plt.ylabel('Word')
			plt.title('Top 20 Words: Neutral Sentiment', fontsize=20);

			most_common_pro = list(itertools.chain
									(*[tweet.split() for tweet in
									   pro['message'].values]))
			freq = FreqDist(most_common_pro)
			y_label = [x[0] for x in freq.most_common(n=20)]
			x_freq = [x[1] for x in freq.most_common(n=20)]
			freq.most_common(n=20)

			# create plot of most common words for pro sentiment
			plt.figure(figsize=(15, 8))
			sns.barplot(data={'x': x_freq, 'y': y_label}, x='x', y='y')
			plt.xlabel('Frequency')
			plt.ylabel('Word')
			plt.title('Top 20 Words: Pro Sentiment', fontsize=20);

			most_common_pro = list(itertools.chain
									(*[tweet.split() for tweet in
									   pro['message'].values]))
			freq = FreqDist(most_common_pro)
			y_label = [x[0] for x in freq.most_common(n=20)]
			x_freq = [x[1] for x in freq.most_common(n=20)]
			freq.most_common(n=20)

			# create plot of most common words for pro sentiment
			plt.figure(figsize=(15, 8))
			sns.barplot(data={'x': x_freq, 'y': y_label}, x='x', y='y')
			plt.xlabel('Frequency')
			plt.ylabel('Word')
			plt.title('Top 20 Words: Pro Sentiment', fontsize=20);

			#News
			most_common_news = list(itertools.chain
								   (*[tweet.split() for tweet in
									  news['message'].values]))
			freq = FreqDist(most_common_news)
			y_label = [x[0] for x in freq.most_common(n=20)]
			x_freq = [x[1] for x in freq.most_common(n=20)]
			freq.most_common(n=20)

			# create plot of most common words for pro sentiment
			plt.figure(figsize=(15, 8))
			sns.barplot(data={'x': x_freq, 'y': y_label}, x='x', y='y')
			plt.xlabel('Frequency')
			plt.ylabel('Word')
			plt.title('Top 20 Words: News Sentiment', fontsize=20);
			st.pyplot()

		if selection_info == "Popular Hastags":  # data is hidden if box is unchecked
			anti_htags = common_hashtags(anti, 10)
			neutral_htags = common_hashtags(neutral, 10)
			pro_htags = common_hashtags(pro, 10)
			news_htags = common_hashtags(news, 10)
			combine_hashtag = pd.concat([anti_htags, neutral_htags, pro_htags, news_htags],
										keys=["Anti", "Neutral", "Pro", "News"])
			combine_hashtag.reset_index(0, inplace=True)
			combine_hashtag.columns = ['sentiment', 'hashtag', 'frequency']

			# plot comparison clustered bar chart
			plt.figure(figsize=(12, 8))
			sns.barplot(x='frequency',
						y='hashtag',
						data=combine_hashtag,
						hue='sentiment',
						palette="pastel")
			plt.title('Hashtag Comparison by Sentiment')
			st.pyplot()

		if selection_info == "Popular Usernames":  # data is hidden if box is unchecked
			# initialise dataframes
			anti_mentions = common_mentions(anti, 10)
			neutral_mentions = common_mentions(neutral, 10)
			pro_mentions = common_mentions(pro, 10)
			news_mentions = common_mentions(news, 10)
			combined_mentions = pd.concat([anti_mentions,
										   neutral_mentions,
										   pro_mentions,
										   news_mentions],
										  keys=["Anti", "Neutral", "Pro", "News"])
			combined_mentions.reset_index(0, inplace=True)
			combined_mentions.columns = ['sentiment', 'username', 'frequency']

			# create username comparison plot
			plt.figure(figsize=(12, 8))
			sns.barplot(x='frequency',
						y='username',
						data=combined_mentions,
						hue='sentiment',
						palette="pastel")
			plt.title('Username Mention Comparison by Sentiment')
			st.pyplot()

		if selection_info == "Popular Emojis":  # data is hidden if box is unchecked
			emoji_appearance = raw[raw.astype(str)['emoji'] != '[]']
			emoji_appearance.sentiment.value_counts()
			emoji = list(itertools.chain(*raw['emoji'].values))
			freq = FreqDist(emoji)
			freq.most_common(n=15)
			# create plot of tweets containing emojis per class
			emoji_classes = ['Pro', 'Neutral', 'Anti', 'News']
			emoji_class_freq = list(emoji_appearance.sentiment.value_counts())
			plt.figure(figsize=(10, 5))
			sns.barplot(x=emoji_classes,
						y=emoji_class_freq,
						palette="pastel")
			plt.xlabel('Sentiment Class')
			plt.ylabel('Tweets with Emojis')
			plt.title('Emojis per Sentiment Class');

			emoji_tweet_order = [len(pro), len(neutral), len(anti), len(news)]
			plt.figure(figsize=(10, 5))
			emoji_fpt_by_class = [x / y for x, y in zip(emoji_class_freq, emoji_tweet_order)]
			sns.barplot(x=emoji_classes,
						y=emoji_fpt_by_class,
						palette="husl")
			plt.xlabel('Sentiment Class')
			plt.ylabel('Ratio of Total Tweets to Tweets with Emojis')
			plt.title('Emoji Appearance per No of Tweets by Sentiment Class');

			st.pyplot()



	# Building out the predication page
	if selection == "Prediction":
		st.subheader("Classifying the tweets by the topics: News, Pro and Anti Global Warming, and neutral")
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	if selection == "Home Page":
		st.subheader("Below is a simple a run through of what the api has to offer!")
		st.info('Tutorial on predicion section:')
		file_ = open("resources\imgs\predicion_tutorial.gif", "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()

		st.markdown(
			f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
			unsafe_allow_html=True,
			)

	if selection == "Hello":
		st.subheader("Hello World")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	#prep(raw)
	main()
