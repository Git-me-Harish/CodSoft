import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import os

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Datasets/IRIS.csv")
    return df

# Preprocess data
def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    lb_encoder = LabelEncoder()
    y_encoded = lb_encoder.fit_transform(y)
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, lb_encoder

# Create and train model
@st.cache_resource
def create_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(50, input_dim=4, activation='relu'),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    history = model.fit(X_train, y_train, epochs=30, batch_size=10, 
                        validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
    
    return model, history

# Load and display flower images
def load_flower_image(species):
    base_dir = "flower_images"
    image_paths = {
        "Iris-setosa": os.path.join(base_dir, "Iris-setosa.jpg"),
        "Iris-versicolor": os.path.join(base_dir, "Iris-versicolor.png"),
        "Iris-virginica": os.path.join(base_dir, "Iris-virginica.jpg")
    }
    
    image_path = image_paths.get(species)
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            return image
        except IOError:
            st.warning(f"Error opening image for {species}.")
            return None
    
    st.warning(f"Image for {species} not found.")
    return None

# Streamlit app
def main():
    st.set_page_config(page_title="Iris Flower Classifier", layout="wide")
    
    # Custom CSS to improve UI
    st.markdown("""
    <style>
    .main {
        background-color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stSelectbox {
        background-color: #e1e5eb;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌸 Advanced Iris Flower Classification")
    
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, lb_encoder = preprocess_data(df)
    
    # Create sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Data Overview", "🧠 Model Training", "📊 Model Evaluation", "🔮 Make Prediction"])
    
    if page == "🏠 Data Overview":
        st.header("Data Overview")
        st.write(df.head())
        
        st.subheader("Iris Species Distribution")
        species = df['species'].value_counts()
        fig = px.pie(values=species, names=species.index, title='Iris Species Distribution', hole=0.3)
        st.plotly_chart(fig)
        
        st.subheader("Feature Relationships")
        fig = px.scatter(df, x="petal_length", y="sepal_width", color="species")
        st.plotly_chart(fig)
        
    elif page == "🧠 Model Training":
        st.header("Model Training")
        
        model, history = create_and_train_model(X_train, y_train, X_test, y_test)
        
        st.subheader("Model Architecture")
        st.text(model.summary())
        
        st.subheader("Training History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
        fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
        fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(fig)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy'))
        fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'))
        fig.update_layout(title='Training and Validation Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
        st.plotly_chart(fig)
        
    elif page == "📊 Model Evaluation":
        st.header("Model Evaluation")
        
        model, _ = create_and_train_model(X_train, y_train, X_test, y_test)
        
        # ROC Curve
        y_test_bin = label_binarize(np.argmax(y_test, axis=1), classes=[0, 1, 2])
        y_pred_proba = model.predict(X_test)
        
        fig = go.Figure()
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(3), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            auc_score = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=color, width=2),
                                     name=f'ROC curve of class {i} (area = {auc_score:.2f})'))
        
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=2, dash='dash'),
                                 name='Random'))
        fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig)
        
        # Confusion Matrix
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual"), x=lb_encoder.classes_, y=lb_encoder.classes_,
                           color_continuous_scale='Blues', title='Confusion Matrix')
        st.plotly_chart(fig_cm)
        
        # Classification Report
        report = classification_report(y_true_classes, y_pred_classes, target_names=lb_encoder.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df)
        
    elif page == "🔮 Make Prediction":
        st.header("Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Features")
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.7, 0.1)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.5, 0.1)
        
        with col2:
            st.subheader("Flower Characteristics")
            st.write(f"Sepal Length: {sepal_length} cm")
            st.write(f"Sepal Width: {sepal_width} cm")
            st.write(f"Petal Length: {petal_length} cm")
            st.write(f"Petal Width: {petal_width} cm")
            
            if st.button("Predict", key="predict_button"):
                model, _ = create_and_train_model(X_train, y_train, X_test, y_test)
                input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
                predicted_class = lb_encoder.inverse_transform([np.argmax(prediction)])
                
                st.success(f"The predicted Iris species is: {predicted_class[0]}")
                
                # Display the flower image
                image = load_flower_image(predicted_class[0])
                if image:
                    st.image(image, caption=f"{predicted_class[0]}", width=300)
                
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame(prediction, columns=lb_encoder.classes_)
                st.dataframe(prob_df)
                
                fig = px.bar(x=lb_encoder.classes_, y=prediction[0], labels={'x': 'Species', 'y': 'Probability'})
                fig.update_layout(title='Prediction Probabilities', xaxis_title='Species', yaxis_title='Probability')
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()