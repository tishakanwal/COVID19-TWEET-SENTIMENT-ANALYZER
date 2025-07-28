import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset (FIXED ENCODING)
df = pd.read_csv("/Users/tishakanwal/Desktop/MyGov/Coronavirus Tweets.csv", encoding='ISO-8859-1')

print("✅ Dataset Loaded Successfully!\n")
print(df.info())
print("\nNull Values:\n", df.isnull().sum())

# 2. Drop Irrelevant Columns
df.drop(['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1, inplace=True)

# 3. Drop Null Tweets/Sentiments
df.dropna(subset=['OriginalTweet', 'Sentiment'], inplace=True)

# 4. Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\@[\w]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['CleanTweet'] = df['OriginalTweet'].apply(clean_text)

# 5. Encode Target Variable 
le = LabelEncoder()
df['SentimentEncoded'] = le.fit_transform(df['Sentiment'])

print("\n📊 Encoded Sentiment Mapping:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# 6. Vectorize the Cleaned Tweets
X = df['CleanTweet']
y = df['SentimentEncoded']

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = tfidf.fit_transform(X)

# 7. Handle Imbalance with SMOTE 
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_vec, y)

# 8. Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# 9. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 10. Evaluate Model 
y_pred = model.predict(X_test)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 11. Plot Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 12. Save Model and Vectorizer 
joblib.dump(model, '/Users/tishakanwal/Desktop/MyGov/sentiment_model.pkl')
joblib.dump(tfidf, '/Users/tishakanwal/Desktop/MyGov/tfidf_vectorizer.pkl')
joblib.dump(le, '/Users/tishakanwal/Desktop/MyGov/label_encoder.pkl')

print("\n✅ Model, TF-IDF Vectorizer, and Label Encoder saved successfully!") 