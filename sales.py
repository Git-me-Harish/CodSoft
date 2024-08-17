import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")

# Custom CSS to improve appearance
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.medium-font {
    font-size:20px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("advertising.csv")
    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Make predictions
def predict_sales(model, tv, radio, newspaper):
    return model.predict([[tv, radio, newspaper]])[0]

# Main Streamlit app
def main():
    st.markdown('<p class="big-font">Sales Prediction Dashboard</p>', unsafe_allow_html=True)
    st.write("This dashboard predicts sales based on advertising spend in TV, Radio, and Newspaper.")

    # Load data and train model
    df = load_data()
    model, X_test, y_test = train_model(df)

    # Sidebar for navigation
    page = st.sidebar.selectbox("Navigation", ["Home", "Prediction", "Model Performance", "Data Exploration"])

    if page == "Home":
        st.markdown('<p class="medium-font">Welcome to the Sales Prediction Dashboard</p>', unsafe_allow_html=True)
        st.write("Use the sidebar to navigate to different sections of the dashboard.")
        st.write("- **Prediction**: Make sales predictions based on advertising spend.")
        st.write("- **Model Performance**: View the model's performance metrics and visualizations.")
        st.write("- **Data Exploration**: Explore the dataset and visualize relationships.")

    elif page == "Prediction":
        st.markdown('<p class="medium-font">Sales Prediction</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            tv = st.slider("TV Advertising Budget", float(df['TV'].min()), float(df['TV'].max()), float(df['TV'].mean()))
        with col2:
            radio = st.slider("Radio Advertising Budget", float(df['Radio'].min()), float(df['Radio'].max()), float(df['Radio'].mean()))
        with col3:
            newspaper = st.slider("Newspaper Advertising Budget", float(df['Newspaper'].min()), float(df['Newspaper'].max()), float(df['Newspaper'].mean()))

        if st.button("Predict Sales"):
            sales = predict_sales(model, tv, radio, newspaper)
            st.success(f"Predicted Sales: ${sales:.2f}")

    elif page == "Model Performance":
        st.markdown('<p class="medium-font">Model Performance</p>', unsafe_allow_html=True)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("R-squared Score", f"{r2:.4f}")

        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
                         title='Actual vs Predicted Sales')
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                 mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        poly_features = model.named_steps['polynomialfeatures']
        feature_names = poly_features.get_feature_names_out(['TV', 'Radio', 'Newspaper'])
        coefficients = model.named_steps['linearregression'].coef_

        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefficients)})
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)

        st.markdown('<p class="medium-font">Top 10 Most Important Features</p>', unsafe_allow_html=True)
        fig = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Data Exploration":
        st.markdown('<p class="medium-font">Data Exploration</p>', unsafe_allow_html=True)
        st.write(df.describe())

        st.markdown('<p class="medium-font">Correlation Heatmap</p>', unsafe_allow_html=True)
        fig = px.imshow(df.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="medium-font">Pairplot</p>', unsafe_allow_html=True)
        fig = px.scatter_matrix(df, dimensions=['TV', 'Radio', 'Newspaper', 'Sales'], color='Sales')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()