
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import load_model


# -----------------------------------
# Custom Activation
# -----------------------------------

@tf.keras.utils.register_keras_serializable(
    package="Custom"
)
def custom_activation(x):

    tanh = tf.keras.activations.tanh(x)
    relu = tf.keras.activations.relu(x)

    return tanh + relu * 1.5


# -----------------------------------
# Streamlit Configuration
# -----------------------------------

st.set_page_config(
    page_title="Social Media Emotion Predictor",
    layout="centered"
)


# -----------------------------------
# Load Resources
# -----------------------------------

@st.cache_resource
def load_resources():

    model = load_model(
        "gru_emotion_model_clean.keras",
        custom_objects={
            "custom_activation": custom_activation
        },
        compile=False
    )

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, scaler, label_encoder


model, scaler, label_encoder = load_resources()

st.title("😊 Social Media Emotion Prediction")
st.write("Enter social media usage details below to predict the dominant emotion.")

# Layout for inputs for the 6 numeric features used by the GRU model
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, help="User's age.")
    daily_usage_time = st.number_input("Daily Usage Time (minutes)", min_value=0.0, max_value=300.0, value=90.0, format="%.1f", help="Average daily time spent on social media in minutes.")
    posts_per_day = st.number_input("Posts Per Day", min_value=0.0, max_value=20.0, value=3.0, format="%.1f", help="Average number of posts made per day.")

with col2:
    likes_received_per_day = st.number_input("Likes Received Per Day", min_value=0.0, max_value=200.0, value=30.0, format="%.1f", help="Average number of likes received per day.")
    comments_received_per_day = st.number_input("Comments Received Per Day", min_value=0.0, max_value=50.0, value=10.0, format="%.1f", help="Average number of comments received per day.")
    messages_sent_per_day = st.number_input("Messages Sent Per Day", min_value=0.0, max_value=100.0, value=20.0, format="%.1f", help="Average number of messages sent per day.")

# Preprocess user input
if st.button("Predict Emotion"):
    # Create a DataFrame containing only the 6 numeric features
    numeric_input_data = pd.DataFrame({
        'Age': [float(age)],
        'Daily_Usage_Time (minutes)': [float(daily_usage_time)],
        'Posts_Per_Day': [float(posts_per_day)],
        'Likes_Received_Per_Day': [float(likes_received_per_day)],
        'Comments_Received_Per_Day': [float(comments_received_per_day)],
        'Messages_Sent_Per_Day': [float(messages_sent_per_day)],
    })

    # Scale these numeric features using the pre-fitted StandardScaler
    input_scaled = scaler.transform(numeric_input_data)

    # Expand dimensions to match the GRU model's expected input shape (batch_size, timesteps, features)
    input_gru = np.expand_dims(input_scaled, axis=1)

    # Make prediction - model.predict returns a batch of predictions, so take the first one [0]
    prediction_probabilities = model.predict(input_gru)[0]

    # Determine the predicted class index (0-indexed for the model output)
    predicted_class_index = np.argmax(prediction_probabilities)

    # Get the confidence for the predicted class
    confidence = prediction_probabilities[predicted_class_index]

    st.divider()

    # Display the prediction result
    # Check if the predicted_class_index corresponds to an actual class from label_encoder
    # The to_categorical function might create an extra column at index 0 if the original labels were 0-indexed.
    # The label_encoder.inverse_transform method expects encoded labels that were used during fit.

    # Map model output index to original emotion label using label_encoder
    # Adjusting for potential extra 0-indexed class from to_categorical
    if predicted_class_index < len(label_encoder.classes_):
        predicted_emotion_name = label_encoder.inverse_transform([predicted_class_index])[0]
        st.success(f"Dominant Emotion: **{predicted_emotion_name}** with **{confidence:.2%}** confidence.")
    elif predicted_class_index == len(label_encoder.classes_): # Handle potential +1 shift if label encoder goes 0-5 and model output is 0-6
        # This handles the case where to_categorical might create an extra class for the highest label + 1
        # We need to map this back to the highest label if it's the intent or handle it as an 'unknown' if it's genuinely a dummy class
        # Given the previous discussion, if output is 7 and classes are 6, the 7th index (index 6) might map to actual class 5 (last)
        # Or, more robustly, directly map based on `label_encoder.classes_`
        all_possible_indices = list(range(len(label_encoder.classes_)))
        if predicted_class_index in all_possible_indices:
             predicted_emotion_name = label_encoder.inverse_transform([predicted_class_index])[0]
             st.success(f"Dominant Emotion: **{predicted_emotion_name}** with **{confidence:.2%}** confidence.")
        else:
             st.warning(f"🤔 Prediction is ambiguous or falls into an undetermined category with {confidence:.2%} confidence.")
    else:
        st.warning(f"🤔 Prediction is ambiguous or falls into an undetermined category with {confidence:.2%} confidence.")

    st.subheader("All Emotion Probabilities:")
    # Display all probabilities using label_encoder.classes_
    for i, prob in enumerate(prediction_probabilities):
        if i < len(label_encoder.classes_):
            emotion_name_for_display = label_encoder.inverse_transform([i])[0]
            st.write(f"- {emotion_name_for_display}: {prob:.2%}")
        else:
            # Handle extra output classes if they exist (e.g., if to_categorical created an extra one)
            st.write(f"- Unknown/Extra Class (Index {i}): {prob:.2%}")
