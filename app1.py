import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# --- 1. LOAD MODELS AND PREPROCESSORS ---
# It's good practice to wrap this in a try-except block for robustness
try:
    # Load the trained model
    model = tf.keras.models.load_model('model.h5')

    # Load the encoders and scaler
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model or preprocessor files not found. Please ensure 'model.h5', 'label_encoder_gender.pkl', 'onehot_encoder_geo.pkl', and 'scaler.pkl' are in the same directory.")
    st.stop()


# --- 2. DEFINE THE EXPECTED COLUMN ORDER ---
# This is the most critical part. The order must exactly match the
# columns that the scaler was fitted on in your training notebook.
TRAINING_COLUMNS = [
    'CreditScore',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
    'Geography_France',
    'Geography_Germany',
    'Geography_Spain'
]


# --- 3. STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title('🏦 Customer Churn Prediction')
st.markdown("Enter the customer's details in the sidebar to predict the likelihood of churn.")

st.sidebar.header("Customer Input Features")

# User input from the sidebar for a cleaner look
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92, 35)
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
balance = st.sidebar.number_input('Balance', value=0.0, format="%.2f")
credit_score = st.sidebar.number_input('Credit Score', 300, 850, 650)
estimated_salary = st.sidebar.number_input('Estimated Salary', value=50000.0, format="%.2f")
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('Has Credit Card?', ('Yes', 'No'))
is_active_member = st.sidebar.selectbox('Is Active Member?', ('Yes', 'No'))

# Convert Yes/No to 1/0
has_cr_card_int = 1 if has_cr_card == 'Yes' else 0
is_active_member_int = 1 if is_active_member == 'Yes' else 0


# --- 4. DATA PREPARATION AND PREDICTION ---
if st.sidebar.button('Predict Churn'):
    # Create a DataFrame for the user's input features (excluding geography)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_int],
        'IsActiveMember': [is_active_member_int],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine the main features with the one-hot encoded geography
    combined_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # **THE FIX**: Reorder the columns to match the training data order
    final_input_df = combined_df.reindex(columns=TRAINING_COLUMNS)

    # Scale the final, correctly-ordered data
    input_data_scaled = scaler.transform(final_input_df)

    # Predict churn using the trained model
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # --- 5. DISPLAY RESULTS ---
    st.header("Prediction Result")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")

    with col2:
        if prediction_proba > 0.5:
            st.error("🔴 High Risk: Customer is likely to churn.")
        else:
            st.success("🟢 Low Risk: Customer is not likely to churn.")

    # Show the processed data for debugging/verification
    with st.expander("Show Processed Input Data"):
        st.write("This is the final data sent to the model after encoding and scaling:")
        st.dataframe(pd.DataFrame(input_data_scaled, columns=TRAINING_COLUMNS))
