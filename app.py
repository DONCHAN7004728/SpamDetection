import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Streamlit Page Configuration
st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="centered")

# Load Pretrained Model
model = load_model('SpamDetectModel.h5', compile=False)

# Initialize Tokenizer (Ensure this matches your training tokenizer)
# If you have saved the tokenizer as a file, load it using joblib:
# tokenizer = joblib.load('tokenizer.pkl')
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# App Title
st.title("🚫 Spam Message Detector")
st.markdown("Enter a message below to check if it's spam or not.")

# Text Input from User
user_input = st.text_area("Enter your message here:", height=150)

# Check for Button Click
if st.button("Detect Spam"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Convert Input Text to Sequences
        sequences = tokenizer.texts_to_sequences([user_input])
        
        if not sequences or len(sequences[0]) == 0:
            st.warning("⚠️ The input message is too short or contains no recognizable words.")
        else:
            try:
                # Pad the Sequence to Match Model's Input Size
                padded = pad_sequences(sequences, maxlen=100)

                # Make Prediction
                prediction = model.predict(padded)[0][0]

                # Display Result
                if prediction > 0.5:
                    st.error("🚨 This message is likely **Spam**.")
                else:
                    st.success("✅ This message is **Not Spam**.")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
