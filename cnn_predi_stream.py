import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'cnn_model_finetuned.h5'
model = load_model(model_path)

# Define class labels
CLASS_LABELS = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non-Demented",
    3: "Very Mild Demented"
}



# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.resize(image, (50, 50))  # Resize to match model input
    image = image / 255.0               # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit App
st.title("Dementia Detection with CNN")
st.write("Upload an MRI image to predict the dementia type.")
st.write("- Mild Demented\n- Moderate Demented\n- Non-Demented\n- Very Mild Demented")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize the image to make it smaller and fit well with the layout
    image_resized = cv2.resize(image, (300, 300))

    # Display the uploaded image with a smaller size
    st.image(image_resized, channels="BGR", caption="Uploaded Image", use_column_width=False)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100

    # Display the results with smaller text size
    st.markdown(f"Prediction: {CLASS_LABELS[predicted_class]}",)
    st.markdown(f"### Confidence: {confidence:.2f}%")

    # Recommendations based on the predicted class
    if predicted_class == 0:
        st.markdown("""
        ### Recommendations for Mild Demented:
        - Engage in brain-stimulating activities like puzzles and memory games.
        - Maintain a consistent daily routine to reduce confusion.
        - Prioritize regular physical exercise and a balanced diet.
        - Schedule regular follow-ups with a healthcare professional.
        """)
    elif predicted_class == 1:
        st.markdown("""
        ### Recommendations for Moderate Demented:
        - Seek immediate medical consultation for advanced care options.
        - Ensure a safe living environment with minimal hazards.
        - Provide emotional and physical support to reduce agitation.
        - Maintain a structured routine and involve caregivers actively.
        """)
    elif predicted_class == 2:
        st.markdown("""
        ### Recommendations for Non-Demented:
        - Continue a healthy lifestyle to maintain brain health.
        - Engage in physical activities like walking or yoga regularly.
        - Follow a diet rich in antioxidants and omega-3 fatty acids.
        - Manage stress levels through relaxation techniques.
        """)
    elif predicted_class == 3:
        st.markdown("""
        ### Recommendations for Very Mild Demented:
        - Monitor symptoms and seek early interventions to prevent progression.
        - Practice mental exercises to boost memory and cognitive skills.
        - Maintain social interactions and engage in group activities.
        - Consult a doctor for preventive strategies and symptom tracking.
        """)

st.markdown("\n\nDeveloped for early detection of dementia using deep learning techniques.")
