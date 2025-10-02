import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. Load the Saved Model and Scaler ---
# Use st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_assets():
    """Loads the trained model and scaler from disk."""
    try:
        model = joblib.load('churn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please ensure 'churn_model.joblib' and 'scaler.joblib' are in the same directory.")
        return None, None

model, scaler = load_assets()

# --- 2. Preprocessing Function for New Data ---
def preprocess_new_data(new_data_df):
    """
    Applies the same preprocessing steps to new data as was used in training.
    """
    df = new_data_df.copy()

    # Encode 'Gender'
    # This assumes 'Male' -> 1, 'Female' -> 0, which is standard for LabelEncoder
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # One-Hot Encode 'Geography'
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

    # Define the columns the model was trained on
    required_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary', 'Gender', 'Geography_Germany', 'Geography_Spain'
    ]
    
    # Add any missing one-hot encoded columns and fill with 0
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match the training data order
    df = df[required_columns]
    
    return df.values # Return as a NumPy array for scaling

# --- 3. Streamlit App Interface ---

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Custom CSS for a better look and feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius:10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stMetric {
        background-color: #FFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title('🤖 Customer Churn Prediction App')
st.markdown("This application uses a trained XGBoost model to predict whether a customer is likely to churn based on their details.")

# Sidebar for user inputs
st.sidebar.header('Enter Customer Details')

# Input fields in the sidebar
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650)
geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=35)
tenure = st.sidebar.number_input('Tenure (Years)', min_value=0, max_value=10, value=5)
balance = st.sidebar.number_input('Balance', min_value=0.0, value=120000.0, format="%.2f")
num_of_products = st.sidebar.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.sidebar.selectbox('Has Credit Card?', ['Yes', 'No'])
is_active_member = st.sidebar.selectbox('Is Active Member?', ['Yes', 'No'])
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, value=80000.0, format="%.2f")

# Predict button
if st.sidebar.button('Predict Churn'):
    if model is not None and scaler is not None:
        # Create a DataFrame from the inputs
        new_customer_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
            'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
            'EstimatedSalary': [estimated_salary]
        })

        # Preprocess, scale, and predict
        preprocessed_data = preprocess_new_data(new_customer_data)
        new_data_scaled = scaler.transform(preprocessed_data)
        
        churn_probability = model.predict_proba(new_data_scaled)[0][1]
        
        # Displaying the results
        st.subheader('Prediction Result')
        
        # Use columns for a cleaner layout
        col1, col2 = st.columns(2)
        
        # Based on analysis, a threshold of ~0.4 provides a good balance.
        # This can be adjusted based on the business goal.
        optimal_threshold = 0.5
        
        if churn_probability >= optimal_threshold:
            col1.error("Prediction: Customer is likely to **Churn**")
            col2.metric(label="Churn Probability", value=f"{churn_probability:.2%}", delta="High Risk")
            st.warning("Action Recommended: Consider initiating retention strategies for this customer.")
        else:
            col1.success("Prediction: Customer is **Not** likely to churn")
            col2.metric(label="Churn Probability", value=f"{churn_probability:.2%}", delta="- Low Risk")
            st.info("No immediate action required. Monitor customer activity as usual.")
            
        # 
        st.markdown("---")
        st.markdown("The prediction is based on the probability score calculated by the model. A higher score indicates a greater likelihood of churn.")
    else:
        st.error("Model is not loaded. Cannot make a prediction.")
