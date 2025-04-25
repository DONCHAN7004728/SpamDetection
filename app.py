import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

model = load_model('SpamDetectModel.h5')
tokenizer = joblib.load('tokenizer.pkl')

st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="centered")
st.title("📩 SMS/Message Spam Detector")
st.markdown("Enter a message below to check if it's **Spam** or **Not Spam**.")

user_input = st.text_area("✉️ Message Text", height=150)

if st.button("🔍 Check Spam"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=7735)

        prediction = model.predict(padded)[0][0]
        label = "🚫 Spam" if prediction > 0.5 else "✅ Not Spam"

        st.markdown(f"### Prediction: **{label}**")
        st.progress(float(prediction) if prediction <= 1 else 1)
