
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🐶🐱 Cats vs Dogs Classifier")

model = tf.keras.models.load_model("cats_vs_dogs_model.h5")

uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = image.resize((160, 160))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 160, 160, 3)
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = round(prediction if prediction > 0.5 else 1 - prediction, 2)
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100}%")
