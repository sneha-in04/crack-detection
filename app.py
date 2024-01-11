import streamlit as st
import tensorflow as tf
from PIL import Image,ImageOps
# Loading Image using PIL
im = Image.open('/Users/snehagupta/Downloads/ml_model/1.png')
# Adding Image to web app
st.set_page_config(page_title="crack detection ", page_icon = im)

def load_model():
    model = tf.keras.models.load_model('/Users/snehagupta/Downloads/model.h5')
    return model

model = load_model()
st.write("# Crack Detection")

file = st.file_uploader("Please upload a surface image", type=["jpg", "png"])

import numpy as np

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size)
    img = np.array(image) / 255
    prediction = model.predict(np.expand_dims(img, axis=0))
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    result = "No crack detected" if predictions < 0.5 else "It has a crack"
    new_title = '<p style="font-family:Times New Roman; color:red; font-size: 30px;">PREDICTION</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.write(f"Prediction: {result} (accuracy: {predictions[0][0]:.2f})")