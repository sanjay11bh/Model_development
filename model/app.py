import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error, f1_score
from model1 import Models, ReadingData, check_best_model  

reader = ReadingData()
model_instance = Models()

def plot_data(data):
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    st.pyplot(plt)

def train_model(X_train, X_test, y_train, y_test, model_name):
    if model_name == 'Linear Regression':
        model, predictions, r2, mae, mse, rmse = model_instance.linear_regressor(X_train, y_train)
        st.write(f"R²: {r2}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")
        
    elif model_name == 'Logistic Regression':
        model, predictions, acc, f1 = model_instance.logistic_regression(X_train, y_train)
        st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")
        st.write(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")

# Streamlit app UI
st.title("Model Training and Prediction App")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = reader.csv_file(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        data = reader.xlsx_file(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(data.head())
    plot_data(data)
    
    option = st.radio("Select an option:", ["Choose a model manually", "Find best model (100 iterations)"])

    if option == "Choose a model manually":
        model_name = st.selectbox("Choose a model", ["Linear Regression", "Logistic Regression"])
        target_column = st.selectbox("Choose target column", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if st.button("Train Model"):
            train_model(X_train, X_test, y_train, y_test, model_name)
    
    elif option == "Find best model (100 iterations)":
        target_column = st.selectbox("Choose target column", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        check_model_instance = check_best_model(model_class=model_instance, X_train=X, y_train=y)
        
        param_bounds_regression = {
            'max_depth': (1, 20),
            'min_samples_split': (2, 10),
            'n_estimators': (50, 200),
            'learning_rate': (0.01, 1.0),
            'max_samples': (0.1, 1.0),
        }
        models_to_evaluate_regression = [
            "Random Forest Regression", "Decision Tree Regression", "Linear Regression"
        ]

        param_bounds_classification = {
            'C': (0.01, 10),
            'degree': (2, 5),
            'max_depth': (1, 20),
            'min_samples_split': (2, 10),
            'n_estimators': (50, 200),
            'learning_rate': (0.01, 1.0),
            'max_samples': (0.1, 1.0),
        }
        models_to_evaluate_classifier = [
            'random_forest_classifier', 'adaboost_classifier', 'bagging_classifier', 'gradient_boosting_classifier'
        ]

        if st.button("Run Model Evaluation Regression"):
            best_score = check_model_instance.evaluate(model_type=models_to_evaluate_regression, params=param_bounds_regression, X=X, y=y)
            st.success(f"Best score: {best_score}")
        
        if st.button("Run Model Evaluation Classification"):
            best_score = check_model_instance.evaluate(model_type=models_to_evaluate_classifier, params=param_bounds_classification, X=X, y=y)
            st.success(f"Best score: {best_score}")
