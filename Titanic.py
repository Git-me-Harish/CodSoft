import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        X_['FamilySize'] = X_['SibSp'] + X_['Parch'] + 1
        X_['IsAlone'] = (X_['FamilySize'] == 1).astype(int)
        X_['Title'] = X_['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        X_['Title'] = X_['Title'].replace(rare_titles, 'Rare')
        X_['Title'] = X_['Title'].replace(['Mlle', 'Ms'], 'Miss')
        X_['Title'] = X_['Title'].replace('Mme', 'Mrs')
        return X_

# Load the model and original dataset
model = joblib.load('Models/titanic_model.joblib')
original_data = pd.read_csv("Datasets/Titanic-Dataset.csv")

def predict_survival(features):
    df = pd.DataFrame([features])
    df['Name'] = 'Mr. John Doe'  # Dummy name for feature engineering
    
    # Feature engineering
    fe = FeatureEngineer()
    df = fe.transform(df)
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    # Ensure all expected columns are present
    expected_columns = getattr(model, 'feature_names_in_', None)
    if expected_columns is None:
        # If model doesn't have feature_names_in_, use all columns
        expected_columns = df.columns
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match the model's expectations
    df = df[expected_columns]
    
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    return prediction[0], probability

def plot_feature_importance():
    # Get feature importances from the model
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        st.warning("Feature importance information is not available for this model.")
        return None
    
    # Get feature names
    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    # Create a DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                 title='Top 10 Feature Importances')
    return fig

def plot_survival_stats():
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Survival by Class", "Survival by Sex", 
                                                        "Survival by Embarkation", "Age Distribution"))
    
    # Survival by Class
    class_survival = original_data.groupby('Pclass')['Survived'].mean().reset_index()
    fig.add_trace(go.Bar(x=class_survival['Pclass'], y=class_survival['Survived'], name='Class'),
                  row=1, col=1)
    
    # Survival by Sex
    sex_survival = original_data.groupby('Sex')['Survived'].mean().reset_index()
    fig.add_trace(go.Bar(x=sex_survival['Sex'], y=sex_survival['Survived'], name='Sex'),
                  row=1, col=2)
    
    # Survival by Embarkation
    embark_survival = original_data.groupby('Embarked')['Survived'].mean().reset_index()
    fig.add_trace(go.Bar(x=embark_survival['Embarked'], y=embark_survival['Survived'], name='Embarkation'),
                  row=2, col=1)
    
    # Age Distribution
    fig.add_trace(go.Histogram(x=original_data[original_data['Survived']==1]['Age'], name='Survived'),
                  row=2, col=2)
    fig.add_trace(go.Histogram(x=original_data[original_data['Survived']==0]['Age'], name='Did not survive'),
                  row=2, col=2)
    
    fig.update_layout(height=800, title_text="Titanic Survival Statistics")
    return fig

st.set_page_config(page_title="Advanced Titanic Survival Predictor", layout="wide")

st.title('Advanced Titanic Survival Prediction')

# Sidebar for user input
st.sidebar.header('Passenger Information')
pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
sex = st.sidebar.radio('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0, 100, 30)
sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
fare = st.sidebar.number_input('Fare', min_value=0.0, max_value=512.0, value=32.2)
embarked = st.sidebar.selectbox('Embarkation Point', ['C', 'Q', 'S'])

features = {
    'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp,
    'Parch': parch, 'Fare': fare, 'Embarked': embarked
}

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if st.button('Predict Survival'):
        prediction, probability = predict_survival(features)
        
        st.write("### Prediction Result:")
        if prediction == 1:
            st.success(f'This passenger would likely **survive** with a probability of {probability:.2f}')
        else:
            st.error(f'This passenger would likely **not survive** with a probability of {1-probability:.2f}')
        
        st.write("### Interpretation:")
        st.write(f"- Passenger Class: {'Lower' if pclass == 3 else 'Middle' if pclass == 2 else 'Upper'} class")
        st.write(f"- Sex: {sex.capitalize()}")
        st.write(f"- Age: {age} years old")
        st.write(f"- Family members aboard: {sibsp + parch}")
        st.write(f"- Fare: ${fare:.2f}")
        st.write(f"- Embarked from: {dict({'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})[embarked]}")

with col2:
    st.subheader("Feature Importance")
    fig = plot_feature_importance()
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Titanic Survival Statistics")
st.plotly_chart(plot_survival_stats(), use_container_width=True)

st.sidebar.markdown("""
This advanced app predicts the probability of a passenger surviving the Titanic disaster based on their characteristics.

Enter the passenger details on the left and click 'Predict Survival' to see the result.

The app also provides visualizations of feature importance and general survival statistics from the Titanic dataset.
""")

