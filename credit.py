import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Function to set the theme
def set_theme(theme):
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            color: #FFFFFF;
            background-color: #4CAF50;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            color: #FFFFFF;
            background-color: #333333;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stButton>button {
            color: #FFFFFF;
            background-color: #008CBA;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            color: #000000;
            background-color: #F0F0F0;
        }
        </style>
        """, unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('Datasets/creditcard_data.csv')
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)
    return data

# Train model
@st.cache_resource
def train_model(data):
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test

# Streamlit app
def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
    
    # Sidebar for theme selection
    theme = st.sidebar.radio("Choose Theme", ["light", "dark"])
    set_theme(theme)
    
    st.title("Credit Card Fraud Detection")
    st.write("This application uses machine learning to detect fraudulent credit card transactions.")
    
    # Load data and train model
    data = load_data()
    model, X_train, y_train, X_test, y_test = train_model(data)
    
    # Display model performance
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{accuracy_score(model.predict(X_train), y_train):.2%}")
    with col2:
        st.metric("Testing Accuracy", f"{accuracy_score(model.predict(X_test), y_test):.2%}")
    
    st.subheader("Fraud Detection")
    st.write("Enter the transaction features to check if it's legitimate or fraudulent.")
    
    # Create input fields for user to enter feature values
    input_df = st.text_input('Input All features (comma-separated)')
    
    # Create a button to submit input and get prediction
    if st.button("Detect Fraud"):
        try:
            input_df_lst = input_df.split(',')
            features = np.array(input_df_lst, dtype=np.float64)
            prediction = model.predict(features.reshape(1,-1))
            
            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.success("Legitimate Transaction")
            else:
                st.error("Fraudulent Transaction")
        except:
            st.error("Please enter valid input. Ensure all features are provided and separated by commas.")

    # Add some information about the features
    st.subheader("Feature Information")
    st.write("This model uses various transaction features to make its prediction. These features include:")
    st.write("- Time: Number of seconds elapsed between this transaction and the first transaction in the dataset")
    st.write("- Amount: Transaction amount")
    st.write("- V1-V28: Anonymized features resulting from a PCA transformation")

    # Add a note about responsible use
    st.info("Note: This is a demo application. In a real-world scenario, always consult with financial experts and use more comprehensive fraud detection systems.")

if __name__ == "__main__":
    main()