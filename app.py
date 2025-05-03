import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set page config for a modern look
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        max-width: 800px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        box-sizing: border-box;
        outline: none;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-top: 20px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        box-sizing: border-box;
        overflow: visible;
        color: #333; /* Ensure text is dark for visibility */
    }
    .prediction-box strong {
        color: #333; /* Ensure strong tags are dark */
    }
    .title {
        color: #ffffff; /* White font for heading */
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2); /* Shadow for readability */
    }
    .subtitle {
        color: #444; /* Darker for better readability */
        font-size: 18px;
        font-weight: 500;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model, tokenizer, and max_len
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('next_word_predictor.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('max_len.pkl', 'rb') as handle:
        max_len = pickle.load(handle)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# Function to predict one next word
def predict_next_word(model, tokenizer, max_len, text, num_predictions=1):
    predicted_words = []
    current_text = text.lower().strip()
    for _ in range(num_predictions):
        token_text = tokenizer.texts_to_sequences([current_text])[0]
        if not token_text:  # Handle empty input
            return ["(no prediction)"]
        padded_token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')
        prediction = model.predict(padded_token_text, verbose=0)
        pos = np.argmax(prediction)
        for word, index in tokenizer.word_index.items():
            if index == pos:
                predicted_words.append(word)
                current_text = current_text + " " + word
                break
    return predicted_words

# App layout
st.markdown("<div class='title'>Next Word Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter a phrase and let our AI predict the next word!</div>", unsafe_allow_html=True)

# Input form
with st.form(key='predict_form'):
    input_text = st.text_input("Enter your text:", placeholder="e.g., where is the")
    submit_button = st.form_submit_button(label="Predict")

# Display predictions
if submit_button and input_text:
    with st.spinner("Predicting..."):
        predicted_words = predict_next_word(model, tokenizer, max_len, input_text)
        predicted_word = predicted_words[0] if predicted_words else "(no prediction)"
        st.markdown(f"""
            <div class='prediction-box'>
                <strong>Input:</strong> {input_text}<br>
                <strong>Predicted next word:</strong> {predicted_word}
            </div>
        """, unsafe_allow_html=True)
elif submit_button and not input_text:
    st.error("Please enter some text to predict.")

# Footer
st.markdown("""
    <div style='text-align: center; color: #999; margin-top: 50px;'>
        Powered by Streamlit & TensorFlow | Next Word Predictor © 2025
    </div>
""", unsafe_allow_html=True)