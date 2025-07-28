import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import joblib
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# Load Artifacts 
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

#  Text Cleaning 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\@[\w]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# Sidebar Navigation 
section = st.sidebar.radio("Go to", ["Single Tweet Prediction", "Batch Prediction (CSV)", "EDA & Insights"])

#  Single Tweet Prediction 
if section == "Single Tweet Prediction":
    st.title("🤖 COVID-19 Tweet Sentiment Classifier")
    tweet = st.text_area("Enter a tweet to analyze sentiment")

    if st.button("Predict Sentiment"):
        cleaned = clean_text(tweet)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        probas = model.predict_proba(vec)[0]
        sentiment = le.inverse_transform([pred])[0]

        st.success(f"Predicted Sentiment: **{sentiment}**")

        # Show probabilities
        st.subheader("Prediction Confidence")
        proba_df = pd.DataFrame({"Sentiment": le.classes_, "Probability": probas})
        fig, ax = plt.subplots()
        sns.barplot(x="Probability", y="Sentiment", data=proba_df, palette="viridis", ax=ax)
        st.pyplot(fig)

# Batch Prediction 
elif section == "Batch Prediction (CSV)":
    st.title("📁 Batch Tweet Sentiment Prediction")
    st.markdown("Upload a CSV file with a column named **OriginalTweet**.")

    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        if 'OriginalTweet' not in df.columns:
            st.error("The CSV must contain a column named 'OriginalTweet'.")
        else:
            df['CleanTweet'] = df['OriginalTweet'].apply(clean_text)
            X_vec = tfidf.transform(df['CleanTweet'])
            df['PredictedSentiment'] = le.inverse_transform(model.predict(X_vec))

            st.success("Prediction complete.")
            st.dataframe(df[['OriginalTweet', 'PredictedSentiment']])

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "sentiment_predictions.csv", "text/csv")

            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            df['PredictedSentiment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, shadow=True, ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

# EDA Section 
elif section == "EDA & Insights":
    st.title("📊 Exploratory Data Analysis - COVID-19 Tweets")

    df = pd.read_csv("Coronavirus Tweets.csv", encoding='ISO-8859-1')
    df.dropna(subset=['OriginalTweet', 'Sentiment'], inplace=True)
    df['CleanTweet'] = df['OriginalTweet'].apply(clean_text)
    df['TweetLength'] = df['CleanTweet'].apply(lambda x: len(x.split()))

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    df['Sentiment'].value_counts().plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Tweet Count")
    st.pyplot(fig1)

    st.subheader("Pie Chart of Sentiments")
    fig2, ax2 = plt.subplots()
    df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

    st.subheader("Tweet Length Distribution by Sentiment")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='Sentiment', y='TweetLength', ax=ax3)
    st.pyplot(fig3)

    st.subheader("Word Cloud of Tweets")
    all_words = ' '.join(df['CleanTweet'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.axis("off")
    st.pyplot(fig4)

    st.subheader("Sentiment-wise Word Frequency")
    selected_sentiment = st.selectbox("Choose a sentiment", df['Sentiment'].unique())
    filtered = df[df['Sentiment'] == selected_sentiment]
    text = " ".join(filtered['CleanTweet'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.imshow(wc, interpolation='bilinear')
    ax5.axis("off")
    st.pyplot(fig5)

    st.info("EDA helps in understanding the class distribution, content characteristics, and model insights.")