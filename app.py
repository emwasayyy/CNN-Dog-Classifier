
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Define the path to the saved model
model_path = 'dogClassifierCNNModel.h5'

# Load the trained model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define image dimensions (should match the training dimensions)
img_height, img_width = 128, 128

# Set the title and description of the app
st.title("Dog vs. Cat Image Classifier")
st.write("Upload an image to classify it as a dog or a cat.")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((img_width, img_height)))
    img_array = img_array / 255.0 # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    score = predictions[0] # Get the prediction score

    # Determine the classification result
    if score > 0.5:
        st.success(f"Prediction: Dog (confidence: {score:.2f})")
    else:
        st.success(f"Prediction: Cat (confidence: {1 - score:.2f})")
