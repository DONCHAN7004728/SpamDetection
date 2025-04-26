import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Disable GPU for Deployment (optional)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Streamlit Page Configuration
st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="centered")

# Load Pretrained Model
try:
    model = load_model('SpamDetectModel.h5', compile=False)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()  # Stop execution if model cannot be loaded

# Initialize Tokenizer (Ensure this matches your training tokenizer)
# If your tokenizer is saved as a file, load it using joblib:
# from joblib import load
# tokenizer = load('tokenizer.pkl')
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

# App Title
st.title("🚫 Spam Message Detector")
st.markdown("Enter a message below to check if it's spam or not.")

# Text Input from User
user_input = st.text_area("✉️ Enter your message:", height=150)

# Check for Button Click
if st.button("🔍 Detect Spam"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Log User Input (for debugging)
        print(f"User input received: {user_input}")

        # Convert Input Text to Sequences
        sequences = tokenizer.texts_to_sequences([user_input])
        print(f"Tokenized sequences: {sequences}")

        if not sequences or len(sequences[0]) == 0:
            st.warning("⚠️ The input message contains no recognizable words.")
            print("Error: Empty or invalid sequences.")
        else:
            try:
                # Pad the Sequence to Match Model's Input Size
                padded = pad_sequences(sequences, maxlen=100)
                print(f"Padded sequence: {padded}")

                # Make Prediction
                prediction = model.predict(padded)[0][0]
                print(f"Prediction score: {prediction}")

                # Display Result
                if prediction > 0.5:
                    st.error("🚨 This message is likely **Spam**.")
                else:
                    st.success("✅ This message is **Not Spam**.")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                print(f"Prediction error: {e}")
