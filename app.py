import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="centered")


model = load_model('SpamDetectModel.h5', compile=False)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")


st.title("🚫 Spam Message Detector")
st.markdown("Enter a message below to check if it's spam or not.")

user_input = st.text_area("📩 Enter your message here", height=150)


if st.button("Detect Spam"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        sequences = tokenizer.texts_to_sequences([user_input])
        
        if not sequences or len(sequences[0]) == 0:
            st.warning("Input message is too short or contains no recognizable words.")
        else:
            try:
                padded = pad_sequences(sequences, maxlen=100)

                prediction = model.predict(padded)[0][0]
                
                if prediction > 0.5:
                    st.error("🚨 This message is likely **Spam**.")
                else:
                    st.success("✅ This message is **Not Spam**.")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
