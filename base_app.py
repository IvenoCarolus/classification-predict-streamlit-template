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
news_vectorizer = open("resources/tfid.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

#Creating file object for bootstrap_tags.txt
bootstrap_blocks = open('bootstrap_tags.txt','r')
text_full_string = bootstrap_blocks.readlines()
profile_blocks = open('profile_summary.txt','r')
profile_strings = profile_blocks.readlines()

#list constant containing dir too all images to be loaded in
IMAGE_DIRS = ['resources/imgs/balance_plot.png','resources/imgs/word_plot.png','resources/imgs/hashtag_plot.png',\
			  'resources/imgs/emoji_plot.png','resources/imgs/tweet_length_plot.png','resources/imgs/url_plot.png',\
			  'resources/imgs/username_plot.png','resources/imgs/Karabo.jpg','resources/imgs/Rohini_Jagath.jpg',\
			  'resources/imgs/Iveno_Carolus.png','resources/imgs/Sizwe.png','resources/imgs/Confidence_Ledwaba.jpg',\
			  'resources/imgs/Tumisang_Sentle.jpg','resources/imgs/twitter.png']


#Loads all profile text
def bfind_profile_text(text_full_string):
	bt_block = ''
	return_var = 0
	who_are_they = ''
	dict1 = {}
	for line in text_full_string:
		if '_START\n' in line:
			return_var = 1
			who_are_they = line.split('_')[0][1:]
			continue
		if return_var == 1:
			if '_END\n' in line:
				pass
				return_var = 0
			else:
				dict1[who_are_they] = line
	return dict1

#Loads in bootstrap html block to use in st.markdown. For Raw Data Section
def bfind_raw_data(text_full_string):
	bt_block = ''
	return_var = 0
	for line in text_full_string:
		if '#RAW_DATA\n' == line:
			return_var = 1
			continue
		if return_var == 1:
			if '#RAW_DATA_END\n' == line:
				break
			else:
				bt_block += line
	return bt_block

#Pulls all bootstrap code for profile sections
def bfind_prof(text_full_string):
	bt_block = ''
	return_var = 0
	for line in text_full_string:
		if '#PROF\n' == line:
			return_var = 1
			continue
		if return_var == 1:
			if '#PROF_END\n' == line:
				break
			else:
				bt_block += line
	return bt_block

#Pulls Navbar
def bfind_navbar(text_full_string):
	bt_block = ''
	return_var = 0
	for line in text_full_string:
		if '#NAV\n' == line:
			return_var = 1
			continue
		if return_var == 1:
			if '#NAV_END\n' == line:
				break
			else:
				bt_block += line
	return bt_block

#Pulls home page
def bfind_home(text_full_string):
	bt_block = ''
	return_var = 0
	for line in text_full_string:
		if '#HOME_PAGE\n' == line:
			return_var = 1
			continue
		if return_var == 1:
			if '#HOME_PAGE_END\n' == line:
				break
			else:
				bt_block += line
	return bt_block

#Pulls home page
def bfind_head(text_full_string):
	bt_block = ''
	return_var = 0
	for line in text_full_string:
		if '#HEAD\n' == line:
			return_var = 1
			continue
		if return_var == 1:
			if '#HEAD_END\n' == line:
				break
			else:
				bt_block += line
	return bt_block

#load in local css styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#For local images only!
#This is not a good way of getting images in and causes the app to load pages way too slow
#Making sure we cache this function to load pages faster
#@st.cache(suppress_st_warning=True)
#def load_in_all_plots(dirs):
#	url_dict = {}
#	for directory in dirs:
#		name = directory.split('/')[-1].split('.')[0]
#		file_ = open(directory, "rb")
#		contents = file_.read()
#		url = base64.b64encode(contents).decode("utf-8")
#		url_dict[name] = url
#		file_.close()
#	return url_dict

#extracting tweet emoji
def tweet_get_emojis(df):
    emoji_2d_list = df['message'].values
    emoji_2d_list = extract_emoji(emoji_2d_list)
    df['emoji'] = emoji_2d_list['emoji']
    return df

raw = pd.read_csv("resources/train.csv")

#Setting up bootstrap constants
RAW_DATA = bfind_raw_data(text_full_string)
PROF_STRING = bfind_prof(text_full_string)
NAV_BAR = bfind_navbar(text_full_string)
HOME_PAGE = bfind_home(text_full_string)
HEAD = bfind_head(text_full_string)
names = bfind_profile_text(profile_strings)
#IMG_URLS = load_in_all_plots(IMAGE_DIRS)

local_css('styles/style.css')

st.markdown(HEAD,unsafe_allow_html=True)
# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	st.title('Chirp - Tweet Classifier')
	options = ["Home Page","Prediction", "Data and Insights",'Meet Our Team']
	selection = st.sidebar.selectbox("Choose Option", options)
	bootstrap_block = ''
	print(names)
	# Building out the "Information" page
	if selection == "Data and Insights":
		st.markdown("Below are some visuals, insights gained from visuals and raw data")
		# You can read a markdown file from supporting resources folder
		insights = ['Raw Data','Data Balance','Tweet Length','Word Frequency','Popular Hastags','Popular Usernames',\
					'Popular Emojis']
		selection_info = st.selectbox("Select page", insights)
		if selection_info == "Raw Data":
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$','Raw Data')
			bootstrap_block_1 = bootstrap_block_1.replace('&&','This tweet Classifier was trained on already classified \
														    tweets pertaining to climate change collected between \
														   Apr 27, 2015 and Feb 21, 2018. The collection of this data \
														   was funded by a Canada Foundation for Innovation JELF Grant \
														   to Chris Bauch, University of Waterloo.')
			st.markdown(bootstrap_block_1,unsafe_allow_html=True)

			st.write(raw[['sentiment', 'message']])
		elif selection_info == "Data Balance":
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', 'Data Balance')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "The training data set is unbalanced, with approximately \
													54% of tweets classified as 'Pro' sentiment. Bias toward classes \
													with more observations is an inherent issue of some classifier \
													methods & rebalancing is required")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)
			st.markdown(
					f'<img src="https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/balance_plot.png">',
					unsafe_allow_html=True
					)

		if selection_info == "Tweet Length":  # data is hidden if box is unchecked
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', 'Tweet Length')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "What is the distribution of tweets by sentiment class?")

			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

			#image display
			st.markdown(
				f'<img src="https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/tweet_length_plot.png">',
				unsafe_allow_html=True
			)

			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block.replace('$$', '')
			bootstrap_block_1 = bootstrap_block.replace('&&', "<ul><li>Overall, tweets in the Pro Climate Change belief sentiment \
														generally have more characters and words than other classes \
														 (cluseter of peach hued points).</li><li>News Tweets are generally \
														 short but use longer words than other classes.</li></ul>")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

		if selection_info == "Word Frequency":  # data is hidden if box is unchecked
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', 'Word Frequency')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "A word cloud is a useful visualisation to assess the most \
															common words (high frequency words are larger) present in \
															the tweet corpus with a single glance.")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

			#image display
			st.markdown(
				f'<img src="https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/word_plot.png">',
				unsafe_allow_html=True
			)

			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', '')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "<ul><li>As expected, topical words surrounding the issue of\
			 												Climate Change are the most prevalent</li>\
			 												<li>News Tweets are generally \
															short but use longer words than other classes.</li>\
															<li>These include 'climate', 'change', 'global', 'warming'.\
															Social & Political figures and themes are also significantly\
															 common: 'trump', 'president', 'people', 'government', \
															 'obama', 'donald', & 'human'.</li>\
															 <li>Words such as 'scientist', 'science', 'believe', 'hoax'\
															 , 'fight', 'action', 'real', 'think', 'denier' & 'right' \
															 represent the polarity of views on climate change.</li>\
															 </ul>")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

		if selection_info == "Popular Hastags":  # data is hidden if box is unchecked
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', 'Hashtags Analysis')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "What are the most common hashtags and how do they relate\
			 												to each class?")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

			#getting image
			st.markdown(
				f'<img src="https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/hashtag_plot.png">',
				unsafe_allow_html=True
			)

			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', '')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "<ul><li>A total of 2598 hashtags are present, 1706 are unique.</li>\
						 												<li>The most common hashtags are centred around \
						 												climate change & political themes.</li>\
																		<li>Notable Hashtags:<ol><li>#beforetheflood: \
																		Reference to Climate Change documentary with \
																		Leonardo DiCaprio (Pro)</li><li>#cop22: \
																		Reference to 2016 United Nations Climate Change\
																		Conference</li><li>#parisagreement: Reference \
																		to the Paris Agreement within the United \
																		Nations Framework Convention on Climate Change\
																		signed in 2016.</li><li>#maga</li></ol></li> \
																		<li>From our analysis it is clear that there are\
																		specific hashtags that are associated with each\
																		class.</li></ul>")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

		if selection_info == "Popular Usernames":  # data is hidden if box is unchecked
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', 'Username Analysis')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "What are the most common username mentions and how \
															do they relate to each class?")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

			st.markdown(
				f'<img src="https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/username_plot.png">',
				unsafe_allow_html=True
			)

			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', '')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "<ul><li>A total of 11808 username mentions in total, \
															7141 are unique.</li>\
									 						<li>The most common username mentions include Political \
									 						figures such as Bernie Sanders, Donald Trump & public \
									 						figures active in advocation for Climate Change mitigation \
									 						such as Leornado DiCaprio & Seth Macfarlane. Media outlets \
									 						such as CNN, National Geographic & The New York times also \
									 						feature in the top ten most common username mentions</li>\
									 						</ul>")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

		if selection_info == "Popular Emojis":  # data is hidden if box is unchecked
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', 'Emoji Analysis')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "An emoji is a digital icon used in text & various '\
			 'social media platforms such as Twitter. Emojis vary from facial expressions to objects reflecting places, \
			  gestures, natural elements or even types of weather for example a snowflake. Emojis could potentially \
			   provide an indication of sentiment")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)
			st.markdown(
				f'<img src="https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/emoji_plot.png">',
				unsafe_allow_html=True
			)
			bootstrap_block_1 = RAW_DATA
			bootstrap_block_1 = bootstrap_block_1.replace('$$', 'More points on the Emoji data:')
			bootstrap_block_1 = bootstrap_block_1.replace('&&', "<ul><li>The most common emoji is the 'Face with Tears \
			                                                          of Joy' emoji, one of the most common emoji \
			                                                           between 2014-2018 and is used to show something \
			                                                           as funny or pleasing. This could hint at sarcasm,\
			                                                            humour or meme culture present in tweets.</li>\
												 						<li>'Thinking Face' emoji (third most common) \
												 						represents thinking or questioning belief. \
												 						Fire and Snowflake emojis make the top ten and \
												 						perfectly illustrate the extreme impacts of \
												 						global warming on weather and climate change.\
												 						</li> \
																		<li>Very few emojis are present in the tweets \
																		overall & the  lack of emojis could possibly \
																		attributed to social media platform as twitter \
																		limits characters in tweets, and as these tweets\
																		 are centred around a serious & divisive \
																		 environmental issue, emojis are used less \
																		 frequently</li> \
												 						</ul>")
			st.markdown(bootstrap_block_1, unsafe_allow_html=True)

	# Building out the predication page
	if selection == "Prediction":
		st.subheader("Classifying the tweets by the topics: News, Pro and Anti Global Warming, and neutral")
		st.info("Classifying your custom text with our ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		predictor = joblib.load(open(os.path.join("resources/best_model.pkl"), "rb"))
		model_selector = st.selectbox("LinearSVC - our best model", ['Linear SVC','Logistic Regression','Random Forest'])
		if model_selector == 'Logistic Regression':
			predictor = joblib.load(open(os.path.join("resources/logstic_regression.pkl"), "rb"))
		elif model_selector == 'Random Forest':
			predictor = joblib.load(open(os.path.join("resources/random_forest.pkl"), "rb"))
		elif model_selector == 'Linear SVC':
			predictor = joblib.load(open(os.path.join("resources/best_model.pkl"), "rb"))

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice

			prediction = predictor.predict(vect_text)

			if prediction == 0:
				prediction = 'Neutral'
			elif prediction == -1:
				prediction = 'Anti Climate Change'
			elif prediction == 1:
				prediction = 'Pro Climate Change'
			elif prediction == 2:
				prediction = 'News'

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

	if selection == "Home Page":
		st.markdown(HOME_PAGE.replace('$$','https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw'\
									  '/master/data/imgs/twitter_bird.png').replace('&&','https://github.com/IvenoCarolu'\
				's/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/bargraph1.jpg'), unsafe_allow_html=True)

	if selection == "Meet Our Team":
		st.markdown("From Data Science to Development, our team has the necessary skills, knowledge & innovation to help grow \
		your business or organisation.Let’s Collaborate!")

		data_url_iveno = 'https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/Iveno_Carolus.jpg'
		data_url_ro= 'https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/Rohini_Jagath.jpg'
		data_url_confy = 'https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs//Confidence_Ledwaba.jpg'
		data_url_karabo = 'https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/Karabo.jpg'
		data_url_sizwe = 'https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/Sizwe.png'
		data_url_tumi = 'https://github.com/IvenoCarolus/kaggle_global_warming_sentiment_predict/raw/master/data/imgs/Tumisang_Sentle.jpg'

		bootstrap_block = PROF_STRING
		#bootstrap_block = bootstrap_block.replace('-VINO-', names['VINO'])
		#bootstrap_block = bootstrap_block.replace('-RO-', names['RO'])
		#bootstrap_block = bootstrap_block.replace('-CONFY-', names['CONFY'])
		#bootstrap_block = bootstrap_block.replace('-SIZWE-', names['SIZWE'])
		#bootstrap_block = bootstrap_block.replace('-KARABO-', names['KARABO'])
		#bootstrap_block = bootstrap_block.replace('-TUMISANG-', names['TUMI'])
		bootstrap_block = bootstrap_block.replace('&&', data_url_iveno)
		bootstrap_block = bootstrap_block.replace('$$', data_url_ro)
		bootstrap_block = bootstrap_block.replace('%%', data_url_confy)
		bootstrap_block = bootstrap_block.replace('!!', data_url_tumi)
		bootstrap_block = bootstrap_block.replace('##', data_url_karabo)
		bootstrap_block = bootstrap_block.replace('^^', data_url_sizwe)
		st.markdown(bootstrap_block,unsafe_allow_html=True)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	#prep(raw)
	main()
