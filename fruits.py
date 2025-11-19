import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("fruit_freshness_cnn.h5")

# Labels
labels = ["Fresh", "Rotten"]

st.title("🍎 Fruits Freshness Detection")
st.write("Upload an image of a fruit to predict if it is **Fresh** or **Rotten**.")

uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    result = labels[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    st.subheader(f"Prediction: **{result}**")
    st.write(f"Confidence: {confidence}%")
