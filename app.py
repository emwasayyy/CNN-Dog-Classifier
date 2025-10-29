import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Define the path to the saved model
# NOTE: For Streamlit Cloud deployment, it's best to use @st.cache_resource
# and define the path inside the function.
model_path = 'dogClassifierCNNModel.h5'

# Load the trained model using st.cache_resource for efficiency
@st.cache_resource
def load_cnn_model():
    """Loads the model only once and caches it."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
model = load_cnn_model()

# Define image dimensions (should match the training dimensions)
img_height, img_width = 128, 128

# Set the title and description of the app
st.title("Dog vs. Cat Image Classifier 🐶🐱")
st.write("Upload an image to classify it as a dog or a cat.")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((img_width, img_height)))
    img_array = img_array / 255.0 # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    
    # *** 💡 CRITICAL FIX FOR ERROR 2 STARTS HERE 💡 ***
    # Keras predictions returns a NumPy array (e.g., [[0.98]]).
    # Use .item() to extract the scalar (single number) value.
    score = predictions[0][0].item() # Access the value at [0][0] and convert to a standard float
    
    # Determine the classification result
    if score > 0.5:
        # 'score' is already the confidence for Dog
        st.success(f"Prediction: Dog (confidence: {score:.2f})")
    else:
        # 1 - 'score' is the confidence for Cat
        st.success(f"Prediction: Cat (confidence: {1 - score:.2f})")
