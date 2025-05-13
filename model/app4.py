# app.py
from model4 import Models, optuna_Model, ReadingData
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.title("AutoML App with Optuna and Streamlit")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    reader = ReadingData()
    df = reader.csv_file(uploaded_file)
    st.dataframe(df.head())

    # Select target column
    target_column = st.selectbox("Select the target column", df.columns)

    if "target_column" not in st.session_state:
        st.session_state.target_column = None

    if st.button("Set target"):
        st.session_state.target_column = target_column
        st.success(f"Target column set to: {target_column}")

    # Proceed only if target column is set
    if st.session_state.target_column is not None:
        problem_type = st.radio("Select problem type", ["Regression", "Classification"])

        X = df.drop(columns=[st.session_state.target_column])
        y = df[st.session_state.target_column]

        # Updated train_test_split to accept X and y directly
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

        model_type = st.selectbox("Select model", [
            "Linear Regression", "Logistic Regression", "Lasso Regression",
            "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost"
        ])

        use_optuna = st.checkbox("Use Optuna for hyperparameter tuning")

        if st.button("Run Model"):
            if use_optuna:
                opt_model = optuna_Model()

                if problem_type == "Regression":
                    if model_type == "Linear Regression":
                        model, pred, r2, mae, mse, rmse = opt_model.linear_regressor(X_train, y_train)
                    elif model_type == "Lasso Regression":
                        model, pred, r2, mae, mse, rmse = opt_model.lasso_regression(X_train, y_train)
                    elif model_type == "Decision Tree":
                        model, pred, r2, mae, mse, rmse = opt_model.decision_tree_regression(X_train, y_train)
                    elif model_type == "Random Forest":
                        model, pred, r2, mae, mse, rmse = opt_model.random_forest_regression(X_train, y_train)
                    elif model_type == "Gradient Boosting":
                        model, pred, r2, mae, mse, rmse = opt_model.gradient_boosting_regression(X_train, y_train)
                    elif model_type == "AdaBoost":
                        model, pred, r2, mae, mse, rmse = opt_model.adaboost_regression(X_train, y_train)
                    st.write(f"Model: {model}") 
                    st.write(f"Predictions: {pred}")
                    st.write(f"Confusion Matrix: {model.confusion_matrix(y_test, pred)}")
                    st.write(f"Classification Report: {model.classification_report(y_test, pred)}")
                    st.write(f"ROC AUC: {model.roc_auc_score(y_test, pred)}")
                    st.write(f"ROC Curve: {model.roc_curve(y_test, pred)}")
                    st.write(f"Feature Importances: {model.feature_importances_}")
                    st.write(f"Best Hyperparameters: {model.best_params_}")
                    st.write(f"Accuracy: {acc}")
                    st.write(f"F1 Score: {f1}")


                else:
                    if model_type == "Logistic Regression":
                        model, pred, acc, f1 = opt_model.logistic_regression(X_train, y_train)
                    elif model_type == "Decision Tree":
                        model, pred, acc, f1 = opt_model.decision_tree_classifier(X_train, y_train)
                    elif model_type == "Random Forest":
                        model, pred, acc, f1 = opt_model.random_forest_classifier(X_train, y_train)
                    elif model_type == "Gradient Boosting":
                        model, pred, acc, f1 = opt_model.gradient_boosting_classifier(X_train, y_train)
                    elif model_type == "AdaBoost":
                        model, pred, acc, f1 = opt_model.adaboost_classifier(X_train, y_train)
                    st.write(f"Model: {model}") 
                    st.write(f"Predictions: {pred}")
                    st.write(f"Confusion Matrix: {model.confusion_matrix(y_test, pred)}")
                    st.write(f"Classification Report: {model.classification_report(y_test, pred)}")
                    st.write(f"ROC AUC: {model.roc_auc_score(y_test, pred)}")
                    st.write(f"ROC Curve: {model.roc_curve(y_test, pred)}")
                    st.write(f"Feature Importances: {model.feature_importances_}")
                    st.write(f"Best Hyperparameters: {model.best_params_}")
                    st.write(f"Accuracy: {acc}")
                    st.write(f"F1 Score: {f1}")
                    

            else:
                model_runner = Models()
                if problem_type == "Regression":
                    if model_type == "Linear Regression":
                        model, pred, r2, mae, mse, rmse = model_runner.Linear_regressor(X_train, y_train)
                    elif model_type == "Lasso Regression":
                        model, pred, r2, mae, mse, rmse = model_runner.Lasso_regression(X_train, y_train)
                    elif model_type == "Decision Tree":
                        model, pred, r2, mae, mse, rmse = model_runner.Decision_tree_regression(X_train, y_train)
                    elif model_type == "Random Forest":
                        model , pred , r2 , mae , mse , rmse = model_runner.Random_forest_regression(X_train, y_train)
                    elif model_type == "Gradient Boosting": 
                        model , pred , r2 , mae , mse , rmse = model_runner.Gradient_boosting_regression(X_train, y_train)
                    elif model_type == "AdaBoost":
                        model , pred , r2 , mae , mse , rmse = model_runner.AdaBoost_regression(X_train, y_train)
                    st.write(f"Model: {model}")
                    st.write(f"R² Score: {r2}")
                    st.write(f"MAE: {mae}")
                    st.write(f"RMSE: {rmse}")
                else:
                    if model_type == "Logistic Regression":
                        model, pred, acc, f1 = model_runner.Logistic_regression(X_train, y_train)
                    if model_type == "Decision Tree":
                        model, pred, acc, f1 = model_runner.Decision_tree_classifier(X_train, y_train)
                    elif model_type == "Random Forest":
                        model, pred, acc, f1 = model_runner.Random_forest_classifier(X_train, y_train)
                    elif model_type == "Gradient Boosting":
                        model, pred, acc, f1 = model_runner.Gradient_boosting_classifier(X_train, y_train)
                    elif model_type == "AdaBoost":
                        model, pred, acc, f1 = model_runner.AdaBoost_classifier(X_train, y_train)
                    st.write(f"Model: {model}")
                    st.write(f"Accuracy: {acc}")
                    st.write(f"F1 Score: {f1}")
                    st.write(f"Predictions: {pred}")
                    st.write(f"Confusion Matrix: {model.confusion_matrix(y_test, pred)}")
                    st.write(f"Classification Report: {model.classification_report(y_test, pred)}")
                    st.write(f"ROC AUC: {model.roc_auc_score(y_test, pred)}")
                    st.write(f"ROC Curve: {model.roc_curve(y_test, pred)}")
                    st.write(f"Feature Importances: {model.feature_importances_}")
                    st.write(f"Best Hyperparameters: {model.best_params_}")
