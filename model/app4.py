# app.py
from model4 import Models, optuna_Model, ReadingData
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import optuna.visualization.matplotlib as optuna_vis



st.title("AutoML App with Optuna and Streamlit")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    reader = ReadingData()
    df = reader.csv_file(uploaded_file)
    st.dataframe(df.head())

    # Select target columns
    all_columns = list(df.columns)
    target_columns = st.multiselect("Select one or more target columns", all_columns)

    if "target_columns" not in st.session_state:
        st.session_state.target_columns = []

    if st.button("Set target"):
        st.session_state.target_columns = target_columns
        st.success(f"Target columns set to: {target_columns}")

    # Proceed only if target columns are set
    if st.session_state.target_columns:
        problem_type = st.radio("Select problem type", ["Regression", "Classification"])

        X = df.drop(columns=st.session_state.target_columns)
        y = df[st.session_state.target_columns]

        # Updated train_test_split to accept X and y directly
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        model_type = st.selectbox("Select model", [
            "Linear Regression", "Logistic Regression", "Lasso Regression",
            "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost"
        ])

       # automate_model = st.selectbox("Select automation method")
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
                        model, pred, r2, mae, mse, rmse = opt_model.gradient_boosting_regressor(X_train, y_train)
                    elif model_type == "AdaBoost":
                        model, pred, r2, mae, mse, rmse  = opt_model.adaboost_regressor(X_train, y_train)
                    st.write(f"Model: {model}") 
                    st.write(f"Predictions: {pred}")
                    st.write(f"R² Score: {r2}")
                    st.write(f"Feature Importances: {model.feature_importances_}")
          
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
#                    st.write(f"Confusion Matrix: {model.confusion_matrix(y_test, pred)}")
                    st.write(f"Feature Importances: {model.feature_importances_}")
                    st.write(f"Best Hyperparameters: {model.best_params_}")
                    st.write(f"Accuracy: {acc}")
                    st.write(f"F1 Score: {f1}")
                    # Extract top 20 feature importances
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = X.columns

                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False).head(20)

                        st.subheader("Top 20 Important Features")
                        st.dataframe(importance_df)

                        # Plot
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        plt.figure(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=importance_df)
                        plt.title("Top 20 Important Features")
                        st.pyplot(plt)
                                        
           # elif automate_model :


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
                        model , pred , r2 , mae , mse , rmse = model_runner.Gradient_boosting_regressor(X_train, y_train)
                    elif model_type == "AdaBoost":
                        model , pred , r2 , mae , mse , rmse = model_runner.AdaBoost_regressor(X_train, y_train)
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
                    # st.write(f"Confusion Matrix: {model.confusion_matrix(y_test, pred)}")

                    st.write(f"Feature Importances: {model.feature_importances_}")
