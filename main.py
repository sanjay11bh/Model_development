import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error, f1_score
from Model import Models, ReadingData, check_best_model  

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
        model, predictions, r2, mae, mse, rmse = model_instance.logistic_regression(X_train, y_train)
        st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")
        st.write(f"F1 Score: {f1_score(y_test, predictions, average='weighted')}")

    elif model_name == 'Decision Tree Regression':
        model, predictions, r2, mae, mse, rmse = model_instance.decision_tree_regression(X_train, y_train)
        st.write(f"R²: {r2}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")
        
    elif model_name == 'Random Forest Regression':
        model, predictions, r2, mae, mse, rmse = model_instance.random_forest_regression(X_train, y_train)
        st.write(f"R²: {r2}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")
        
    elif model_name == 'Random Forest Classifier':
        model, predictions, acc, f1 = model_instance.random_forest_classifier(X_train, y_train)
        st.write(f"Accuracy: {acc}")
        st.write(f"F1 Score: {f1}")
        st.write(f"Accuracy (test set): {accuracy_score(y_test, predictions)}")
        st.write(f"F1 Score (test set): {f1_score(y_test, predictions, average='weighted')}")
    
    elif model_name == 'Decision Tree Classifier':
        model, predictions, acc, f1 = model_instance.decision_tree_classifier(X_train, y_train)
        st.write(f"Accuracy: {acc}")
        st.write(f"F1 Score: {f1}")
        st.write(f"Accuracy (test set): {accuracy_score(y_test, predictions)}")
        st.write(f"F1 Score (test set): {f1_score(y_test, predictions, average='weighted')}")

# Streamlit app UI
st.title("Model Training and Prediction App")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load the data based on file type
    if uploaded_file.name.endswith(".csv"):
        data = reader.csv_file(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        data = reader.xlsx_file(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Visualize data
    plot_data(data)

    # Choose method
    option = st.radio("Select an option:", ["Choose a model manually", "Find best model (100 iterations)"])

    if option == "Choose a model manually":
        # Select model
        model_name = st.selectbox("Choose a model", ["Linear Regression", "Logistic Regression", "Decision Tree Regression",
                                                    "Random Forest Regression", "Random Forest Classifier", "Decision Tree Classifier"])
        # Select target column for supervised learning tasks
        target_column = st.selectbox("Choose target column", data.columns)
        
        # Prepare data for model training
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the selected model
        if st.button("Train Model"):
            train_model(X_train, X_test, y_train, y_test, model_name)
    
    elif option == "Find best model (100 iterations)":
        target_column = st.selectbox("Choose target column", data.columns)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Initialize check_best_model with the correct arguments
        # In the Streamlit app
        if st.button("Run Model Evaluation Regression"):
            check_model_instance = check_best_model(model_class=model_instance, X_train=X, y_train=y)

                        # Flatten the param_bounds dictionary to a format compatible with BayesianOptimization
            param_bounds = {
                'degree': (2, 5),  # Bounds for degree of the polynomial for linear regression
                'max_depth': (1, 20),  # Bounds for max_depth for decision tree and random forest models
                'min_samples_split': (2, 10),  # Bounds for min_samples_split for decision tree and random forest
                'n_estimators': (50, 200),  # Bounds for number of trees for random forest, adaboost, and bagging models
                'learning_rate': (0.01, 1.0),  # Bounds for learning_rate for boosting models
                'max_samples': (0.1, 1.0),  # Bounds for max_samples for bagging classifier
                'min_samples_split': (2, 10),  # Bounds for min_samples_split for all tree-based models
                'n_estimators': (50, 200),  # Bounds for number of trees for adaboost, gradient boosting, and bagging
                'learning_rate': (0.01, 0.1),  # Bounds for learning_rate for gradient boosting classifier
                'max_depth': (3, 10),  # Bounds for max_depth for gradient boosting classifier
            }

            # Specify models to evaluate
            models_to_evaluate_regression= [
                'Decision Tree Regression'
            ]

            # Now when calling evaluate, you can pass the flattened param_bounds

            best_score = check_model_instance.evaluate(model_type=models_to_evaluate_regression, params=param_bounds , X= X, y=y)
            st.success(f"Best score: {best_score}")


        elif st.button("Run Model Evaluation Classification"):
            check_model_instance = check_best_model(model_class=model_instance, X_train=X, y_train=y)

                        # Flatten the param_bounds dictionary to a format compatible with BayesianOptimization
            param_bounds = {
                'C': (0.01, 10),  # Bounds for C parameter (regularization strength for linear regression)
                'degree': (2, 5),  # Bounds for degree of the polynomial for linear regression
                'max_depth': (1, 20),  # Bounds for max_depth for decision tree and random forest models
                'min_samples_split': (2, 10),  # Bounds for min_samples_split for decision tree and random forest
                'n_estimators': (50, 200),  # Bounds for number of trees for random forest, adaboost, and bagging models
                'learning_rate': (0.01, 1.0),  # Bounds for learning_rate for boosting models
                'max_samples': (0.1, 1.0),  # Bounds for max_samples for bagging classifier
                'min_samples_split': (2, 10),  # Bounds for min_samples_split for all tree-based models
                'n_estimators': (50, 200),  # Bounds for number of trees for adaboost, gradient boosting, and bagging
                'learning_rate': (0.01, 0.1),  # Bounds for learning_rate for gradient boosting classifier
                'max_depth': (3, 10),  # Bounds for max_depth for gradient boosting classifier
            }

            # Specify models to evaluate

            models_to_evaluate_classifier= [
                'random_forest_classifier',
                'adaboost_classifier',
                'bagging_classifier',
                'gradient_boosting_classifier'
            ]

            # Now when calling evaluate, you can pass the flattened param_bounds
            best_score = check_model_instance.evaluate(model_type=models_to_evaluate_classifier, params=param_bounds, X= X, y=y)
            st.success(f"Best score: {best_score}")






