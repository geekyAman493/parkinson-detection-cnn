import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('parkinson_disease_detection.h5')  # Replace with your model's path

# Define labels
labels = ['Healthy', 'Parkinson']

# Preprocessing function for uploaded images
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Resize to match model input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Streamlit App
st.title("Parkinson's Disease Detection")
st.write("Upload a spiral or wave drawing image to classify it as 'Healthy' or 'Parkinson'.")

# Upload button
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode image from buffer

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    predicted_label = labels[np.argmax(prediction[0])]

    # Display the image and prediction
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated to use_container_width
    st.markdown(f'<h3 style="color: green;">Prediction: {predicted_label}</h3>', unsafe_allow_html=True)

