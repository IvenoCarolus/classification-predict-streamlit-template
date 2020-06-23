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
# Streamlit dependencies
import streamlit as st
import joblib,os,base64
import seaborn as sns

# Data dependencies
import pandas as pd
import numpy as np

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

#load in local css styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	local_css('styles/style.css')
	st.title("Tweet Classifer (Lemmatizers)")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home Page","Prediction", "Information","Hello"]
	selection = st.sidebar.selectbox("Choose Option", options)
	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Raw data and insights gained:")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Data balance")
		if st.checkbox('Show visuals for Data balance'):  # data is hidden if box is unchecked
			#b_data = pd.DataFrame(np.array([len(raw[raw['sentiment'] == -1]),len(raw[raw['sentiment'] == 0]),len(raw[raw['sentiment'] == 1]),len(raw[raw['sentiment'] == 2])]),columns = ["Anti", "Neutral", "Pro","News"])
			balance_display = {'Anti':len(raw[raw['sentiment'] == -1]),'Neutral':len(raw[raw['sentiment'] == 0]),'Pro':len(raw[raw['sentiment'] == 1]),'News':len(raw[raw['sentiment'] == 2])}
			#b_data = pd.DataFrame(list(balance_display.items()))
			sns.barplot(x=list(balance_display.keys()), y=list(balance_display.values()))
			#st.write(balance_display.keys(),balance_display.values())
			st.pyplot()
			#st.bar_chart(data=b_data, width=0, height=0,x=['Anti','Neutral','Pro','News'], use_container_width=True) # will write the df to the page
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
		file_ = open("resources/imgs/predicion_tutorial.gif", "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()

		st.markdown(
			f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
			unsafe_allow_html=True,
			)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
