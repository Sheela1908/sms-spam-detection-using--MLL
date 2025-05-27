import streamlit as st
import pickle
import string
import nltk
# Ensure necessary resources are downloaded
#nltk.download('punkt')
#nltk.download('stopwords')
import os
import nltk

nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# Preprocess
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
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


# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# UI
st.title("📩 SMS Spam Classifier")

input_sms = st.text_input("Enter the Message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")

# # Optional: CSV Upload
# st.markdown("---")
# st.subheader("📁 Batch Prediction (Optional)")
# uploaded_file = st.file_uploader("Upload a CSV file with a `message` column", type=["csv"])
#
# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         if 'message' not in df.columns:
#             st.error("❌ CSV must contain a `message` column.")
#         else:
#             df['transformed'] = df['message'].apply(transform_text)
#             vectorized = tfidf.transform(df['transformed'])
#             df['Prediction'] = model.predict(vectorized)
#             df['Prediction'] = df['Prediction'].map({1: 'Spam', 0: 'Not Spam'})
#             st.write(df[['message', 'Prediction']])
#             csv = df.to_csv(index=False).encode('utf-8')
#             st.download_button("📥 Download Results", data=csv, file_name="spam_predictions.csv", mime='text/csv')
#     except Exception as e:
#         st.error(f"Error processing file: {e}")
