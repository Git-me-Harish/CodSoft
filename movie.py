import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Datasets/IMDb Movies India.csv", encoding='latin-1')  # We can also try different encodings if needed
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    # Convert 'Year' to numeric, removing any non-numeric characters
    df['Year'] = pd.to_numeric(df['Year'].str.extract('(\d+)', expand=False), errors='coerce')
    
    # Convert 'Duration' to numeric, removing 'min' and any other non-numeric characters
    df['Duration'] = pd.to_numeric(df['Duration'].str.extract('(\d+)', expand=False), errors='coerce')
    
    # Ensure 'Votes' is numeric
    df['Votes'] = pd.to_numeric(df['Votes'].str.replace(',', ''), errors='coerce')
    
    # Feature extraction
    df['Genre_mean_rating'] = df.groupby('Genre')['Rating'].transform('mean')
    df['Director_encoded'] = df.groupby('Director')['Rating'].transform('mean')
    df['Actor1_encoded'] = df.groupby('Actor 1')['Rating'].transform('mean')
    df['Actor2_encoded'] = df.groupby('Actor 2')['Rating'].transform('mean')
    df['Actor3_encoded'] = df.groupby('Actor 3')['Rating'].transform('mean')

    # Drop rows with NaN values
    df_cleaned = df.dropna(subset=['Year', 'Votes', 'Duration', 'Genre_mean_rating', 'Director_encoded', 
                                   'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded', 'Rating'])
    
    st.write(f"Rows before cleaning: {len(df)}")
    st.write(f"Rows after cleaning: {len(df_cleaned)}")
    
    return df_cleaned

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(n_estimators=100, random_state=42))
        ])
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

def main():
    st.title("Movie Review Prediction App")

    df = load_data()
    if df is None:
        st.stop()
    
    df_cleaned = preprocess_data(df)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Model Performance", "Predict Rating"])

    if page == "Data Overview":
        st.header("Data Overview")
        st.write(df_cleaned.head())
        st.write(f"Shape of data: {df_cleaned.shape}")

        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature", ['Year', 'Duration', 'Rating', 'Votes'])
        fig = px.histogram(df_cleaned, x=feature, nbins=30, title=f'Distribution of {feature}')
        st.plotly_chart(fig)

        st.subheader("Feature Relationships")
        x_axis = st.selectbox("Select X-axis", ['Duration', 'Votes', 'Year'])
        y_axis = st.selectbox("Select Y-axis", ['Rating', 'Votes', 'Duration'])
        fig = px.scatter(df_cleaned, x=x_axis, y=y_axis, color='Rating', title=f'{y_axis} vs {x_axis}')
        st.plotly_chart(fig)

    elif page == "Model Performance":
        st.header("Model Performance")

        X = df_cleaned[['Year', 'Votes', 'Duration', 'Genre_mean_rating', 'Director_encoded', 
                        'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']]
        y = df_cleaned['Rating']

        models, X_test, y_test = train_models(X, y)

        for name, model in models.items():
            st.subheader(f"{name} Performance")
            mse, rmse, mae, r2 = evaluate_model(model, X_test, y_test)
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"Root Mean Squared Error: {rmse:.4f}")
            st.write(f"Mean Absolute Error: {mae:.4f}")
            st.write(f"R-squared Score: {r2:.4f}")

    elif page == "Predict Rating":
        st.header("Predict Movie Rating")

        year = st.number_input("Year", min_value=1900, max_value=2023, value=2000)
        votes = st.number_input("Votes", min_value=0, value=1000)
        duration = st.number_input("Duration (minutes)", min_value=0, value=120)
        genre_mean_rating = st.number_input("Genre Mean Rating", min_value=0.0, max_value=10.0, value=7.0)
        director_encoded = st.number_input("Director Encoded", min_value=0.0, max_value=10.0, value=7.0)
        actor1_encoded = st.number_input("Actor 1 Encoded", min_value=0.0, max_value=10.0, value=7.0)
        actor2_encoded = st.number_input("Actor 2 Encoded", min_value=0.0, max_value=10.0, value=7.0)
        actor3_encoded = st.number_input("Actor 3 Encoded", min_value=0.0, max_value=10.0, value=7.0)

        input_data = np.array([[year, votes, duration, genre_mean_rating, director_encoded, 
                                actor1_encoded, actor2_encoded, actor3_encoded]])

        if st.button("Predict Rating"):
            X = df_cleaned[['Year', 'Votes', 'Duration', 'Genre_mean_rating', 'Director_encoded', 
                            'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']]
            y = df_cleaned['Rating']

            models, _, _ = train_models(X, y)
            xgb_model = models['XGBoost']

            prediction = xgb_model.predict(input_data)
            st.success(f"Predicted Rating: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()