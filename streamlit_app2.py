import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Configuration ---
st.set_page_config(
    page_title="COVID-19 Tweet Sentiment Classifier",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Dark Theme Styling ---
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0E1117;
            color: #CFCFCF;
        }
        .stTextInput > div > div > input, .stTextArea > div > textarea {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .stFileUploader > div > div > div > button {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model and vectorizer ---
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Label Mapping ---
label_map = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

# --- Sample tweets ---
sample_tweets = [
    "Feeling blessed after getting the vaccine today! #CovidVaccine #Hope",
    "The number of cases keeps rising. We are doomed. #COVID19",
    "Today the government announced 30,000 new COVID-19 cases.",
    "I'm exhausted. Mentally and physically drained because of this pandemic.",
    "Huge thanks to all frontline workers working 24/7 to keep us safe."
]

# --- Header ---
st.title("🦠 COVID-19 Tweet Sentiment Classifier")
st.markdown("This app classifies the sentiment of COVID-19 related tweets using a trained machine learning model.")

# --- Sidebar: Input mode ---
with st.sidebar:
    st.header("🔧 Input Type")
    input_mode = st.radio("Choose input method", ["✍️ Manual Text", "🧪 Sample Tweet", "📂 Upload CSV"])
    st.markdown("---")
    st.caption("Developed by Tisha Kanwal • MyGov Internship Project")

# --- Helper function for prediction ---
def predict_sentiment(text):
    input_vec = vectorizer.transform([text])
    pred_class = model.predict(input_vec)[0]
    pred_probs = model.predict_proba(input_vec)[0]
    pred_label = label_map.get(pred_class, str(pred_class))
    return pred_label, pred_probs

# --- Manual Text Input ---
if input_mode == "✍️ Manual Text":
    tweet_text = st.text_area("Enter your tweet:", placeholder="e.g., I feel hopeful after getting the vaccine.")
    if st.button("🔍 Analyze"):
        if tweet_text.strip() == "":
            st.warning("⚠️ Please enter a tweet.")
        else:
            pred_label, pred_probs = predict_sentiment(tweet_text)
            sentiments = [label_map.get(cls, str(cls)) for cls in model.classes_]

            # Results
            st.subheader("🎯 Prediction Result")
            st.success(f"➡️ Predicted Sentiment: **{pred_label}**")

            st.subheader("📊 Model Confidence")
            conf_df = pd.DataFrame({
                "Sentiment": sentiments,
                "Confidence (%)": np.round(pred_probs[:len(sentiments)] * 100, 2)
            })
            st.dataframe(conf_df)

            # Pie Chart
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(pred_probs[:len(sentiments)], labels=sentiments, autopct='%1.1f%%',
                   startangle=90, colors=["#e57373", "#ffb74d", "#81c784"])
            ax.axis("equal")
            st.pyplot(fig)

# --- Sample Tweet Input ---
elif input_mode == "🧪 Sample Tweet":
    tweet_text = st.selectbox("Select a sample tweet:", sample_tweets)
    if st.button("🔍 Analyze"):
        pred_label, pred_probs = predict_sentiment(tweet_text)
        sentiments = [label_map.get(cls, str(cls)) for cls in model.classes_]

        st.subheader("🎯 Prediction Result")
        st.success(f"➡️ Predicted Sentiment: **{pred_label}**")

        st.subheader("📊 Model Confidence")
        conf_df = pd.DataFrame({
            "Sentiment": sentiments,
            "Confidence (%)": np.round(pred_probs[:len(sentiments)] * 100, 2)
        })
        st.dataframe(conf_df)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(pred_probs[:len(sentiments)], labels=sentiments, autopct='%1.1f%%',
               startangle=90, colors=["#e57373", "#ffb74d", "#81c784"])
        ax.axis("equal")
        st.pyplot(fig)

# --- CSV Upload ---
elif input_mode == "📂 Upload CSV":
    st.markdown("Upload a CSV file. We'll try to detect the column with tweet text.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines='skip')
            # Try common column names
            possible_columns = ['tweet', 'Tweet', 'text', 'Text', 'OriginalTweet']
            text_column = None
            for col in possible_columns:
                if col in df.columns:
                    text_column = col
                    break

            if text_column is None:
                st.error("❌ Could not detect a column with tweet-like text.")
            else:
                st.success(f"✅ Detected tweet column: `{text_column}`")
                if st.button("🔍 Analyze All Tweets"):
                    df["Predicted Sentiment"] = model.predict(vectorizer.transform(df[text_column]))

                    st.subheader("📄 Prediction Results")
                    st.dataframe(df[[text_column, "Predicted Sentiment"]])

                    sentiment_counts = df["Predicted Sentiment"].value_counts()
                    labels = [label_map.get(s, str(s)) for s in sentiment_counts.index]
                    sizes = sentiment_counts.values

                    fig, ax = plt.subplots()
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                           startangle=90, colors=["#e57373", "#ffb74d", "#81c784"])
                    ax.axis("equal")
                    st.subheader("📊 Sentiment Distribution")
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
