import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Real-Time Text Analysis", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv(
        "https://raw.githubusercontent.com/dair-ai/emotion_dataset/master/data/train.txt",
        sep=";",
        names=["text","label"]
    )

data = load_data()
texts = data["text"].astype(str)
labels = data["label"]

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title("Real-Time Text Sentiment Analyzer")
st.write("Accuracy:", round(acc, 3))

st.subheader("Live Text Input")

user_text = st.text_area("Enter text here")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([user_text])
        prediction = model.predict(vec)[0]
        st.success(f"Prediction: {prediction}")

st.subheader("Real-Time Stream Simulation")

sample_texts = [
    "I am very happy today",
    "This is terrible",
    "I feel neutral about this",
    "I am scared and anxious"
]

if st.button("Start Stream"):
    placeholder = st.empty()
    for t in sample_texts:
        vec = vectorizer.transform([t])
        pred = model.predict(vec)[0]
        placeholder.info(f"{t} → {pred}")
        time.sleep(1)
