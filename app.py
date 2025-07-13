import streamlit as st
import pickle
import string
import nltk
import re
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# --- Ensure required NLTK resource is available ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()

# --- Preprocessing function ---
def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for i in text:
        if re.match(r'https?://\S+|www\.\S+|\S+\.(com|net|org|co|uk|in|info|biz)', i):
            y.append('_URL_')
        elif re.match(r'(£|\$|€)?\d{3,}', i):
            y.append('_NUM_')
        elif re.match(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', i):
            y.append('_EMAIL_')
        elif i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --- Load model and vectorizer ---
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model files 'model.pkl' or 'vectorizer.pkl' not found.")
    st.stop()

try:
    tfidf = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# --- Page config ---
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom dark theme CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        background-color: #2D2D2D;
        color: white;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #ff1c1c;
    }
    </style>
""", unsafe_allow_html=True)

# --- App UI ---
st.title("📩 SMS Spam Classifier")
st.markdown("Instantly check whether an SMS is Spam or Not")
st.markdown("---")

input_sms = st.text_area("✉️ Enter your SMS message below:", height=150, placeholder="Type or paste your SMS message here...")

if st.button('🔍 Classify'):
    if input_sms.strip():
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        if hasattr(model, 'predict_proba'):
            spam_proba = model.predict_proba(vector_input)[0][1]
            threshold = 0.5  # Adjust threshold here
            if spam_proba >= threshold:
                st.error("🚫 Spam")
            else:
                st.success("✅ Not Spam")
        else:
            result = model.predict(vector_input)[0]
            st.error("🚫 Spam" if result == 1 else "✅ Not Spam")
    else:
        st.warning("⚠️ Please enter a message to classify.")

st.markdown("---")
st.caption("🔐 This classifier uses machine learning and NLP to detect SMS spam messages.")
