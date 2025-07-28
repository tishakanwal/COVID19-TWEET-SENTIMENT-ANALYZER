# File 1: covid19_tweet_sentiments.py
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset 
df = pd.read_csv("/Users/tishakanwal/Desktop/MyGov/Coronavirus Tweets.csv", encoding='ISO-8859-1')
print("\n✅ Dataset Loaded Successfully!")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Understanding 
print("\nSentiment Distribution:")
print(df['Sentiment'].value_counts())

sns.countplot(data=df, x='Sentiment', palette='Set2')
plt.title("Distribution of Sentiment Labels")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#  3. Drop Irrelevant Columns
df.drop(['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1, inplace=True)
df.dropna(subset=['OriginalTweet', 'Sentiment'], inplace=True)

# 4. Text Cleaning 
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\@[\w]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['CleanTweet'] = df['OriginalTweet'].apply(clean_text)
df['TweetLength'] = df['OriginalTweet'].apply(lambda x: len(str(x).split()))

sns.histplot(df['TweetLength'], bins=30, kde=True)
plt.title("Tweet Length Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 5. Encode Target 
le = LabelEncoder()
df['SentimentEncoded'] = le.fit_transform(df['Sentiment'])

print("\nEncoded Sentiment Mapping:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# 6. TF-IDF Vectorization 
X = df['CleanTweet']
y = df['SentimentEncoded']

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 7. Handle Imbalance 
sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.show()

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_vec, y)

sns.countplot(x=y_res)
plt.title("Class Distribution After SMOTE")
plt.show()

# ---------------------- 8. Train-Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ---------------------- 9. Model Training ----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------- 10. Evaluation ----------------------
y_pred = model.predict(X_test)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n📊 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Macro Precision:", precision_score(y_test, y_pred, average='macro'))
print("Macro Recall:", recall_score(y_test, y_pred, average='macro'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ---------------------- 11. Feature Importance ----------------------
feature_names = vectorizer.get_feature_names_out()
for i, class_label in enumerate(le.classes_):
    top_features = np.argsort(model.coef_[i])[-10:]
    print(f"\nTop features for sentiment '{class_label}':")
    print([feature_names[j] for j in top_features])

# ---------------------- 12. Save Artifacts ----------------------
joblib.dump(model, '/Users/tishakanwal/Desktop/MyGov/sentiment_model.pkl')
joblib.dump(vectorizer, '/Users/tishakanwal/Desktop/MyGov/tfidf_vectorizer.pkl')
joblib.dump(le, '/Users/tishakanwal/Desktop/MyGov/label_encoder.pkl')

print("\n✅ Model, TF-IDF Vectorizer, and Label Encoder saved successfully!")