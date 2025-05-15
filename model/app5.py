from model4 import Models, optuna_Model, ReadingData, AutoModelSelector
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import optuna.visualization.matplotlib as optuna_vis
import os

st.title("AutoML App with Optuna and Streamlit")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    reader = ReadingData()
    df = reader.csv_file(uploaded_file)
    st.dataframe(df.head())

    all_columns = list(df.columns)
    target_columns = st.multiselect("Select one or more target columns", all_columns)

    if "target_columns" not in st.session_state:
        st.session_state.target_columns = []

    if st.button("Set target"):
        st.session_state.target_columns = target_columns
        st.success(f"Target columns set to: {target_columns}")

    if st.session_state.target_columns:
        problem_type = st.radio("Select problem type", ["Regression", "Classification"])

        X = df.drop(columns=st.session_state.target_columns)
        y = df[st.session_state.target_columns]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mode = st.radio("Select mode", ["Manual", "Automatic"])

        use_optuna = st.checkbox("Use Optuna for hyperparameter tuning")

        if mode == "Manual":
            model_type = st.selectbox("Select model", [
                "Linear Regression", "Logistic Regression", "Lasso Regression",
                "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost"
            ])

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
                            model, pred, r2, mae, mse, rmse = opt_model.adaboost_regressor(X_train, y_train)

                        st.write(f"R² Score: {r2}")
                        st.write(f"MAE: {mae}")
                        st.write(f"RMSE: {rmse}")
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

                        st.write(f"Accuracy: {acc}")
                        st.write(f"F1 Score: {f1}")

                    pred_df, file_path = opt_model.predict_and_save(model, X_test)
                    st.dataframe(pred_df)
                    with open(file_path, "rb") as f:
                        st.download_button("Download Predictions CSV", f, file_name="predictions.csv", mime="text/csv")

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
                            model, pred, r2, mae, mse, rmse = model_runner.Random_forest_regression(X_train, y_train)
                        elif model_type == "Gradient Boosting":
                            model, pred, r2, mae, mse, rmse = model_runner.Gradient_boosting_regressor(X_train, y_train)
                        elif model_type == "AdaBoost":
                            model, pred, r2, mae, mse, rmse = model_runner.AdaBoost_regressor(X_train, y_train)

                        st.write(f"R² Score: {r2}")
                        st.write(f"MAE: {mae}")
                        st.write(f"RMSE: {rmse}")
                    else:
                        if model_type == "Logistic Regression":
                            model, pred, acc, f1 = model_runner.Logistic_regression(X_train, y_train)
                        elif model_type == "Decision Tree":
                            model, pred, acc, f1 = model_runner.Decision_tree_classifier(X_train, y_train)
                        elif model_type == "Random Forest":
                            model, pred, acc, f1 = model_runner.Random_forest_classifier(X_train, y_train)
                        elif model_type == "Gradient Boosting":
                            model, pred, acc, f1 = model_runner.Gradient_boosting_classifier(X_train, y_train)
                        elif model_type == "AdaBoost":
                            model, pred, acc, f1 = model_runner.AdaBoost_classifier(X_train, y_train)

                        st.write(f"Accuracy: {acc}")
                        st.write(f"F1 Score: {f1}")

                    pred_df, file_path = model_runner.predict_and_save(model, X_test)
                    st.dataframe(pred_df)
                    with open(file_path, "rb") as f:
                        st.download_button("Download Predictions CSV", f, file_name="predictions.csv", mime="text/csv")

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = X.columns
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False).head(20)

                    st.subheader("Top 20 Important Features")
                    st.dataframe(importance_df)

                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df)
                    plt.title("Top 20 Important Features")
                    st.pyplot(plt)

                st.markdown("---")
                st.subheader("Predict on New Data")

                new_data_file = st.file_uploader("Upload new dataset (no target column)", key="new_data")

                if new_data_file is not None:
                    new_df = pd.read_csv(new_data_file)
                    st.dataframe(new_df.head())

                    if st.button("Predict on New Data"):
                        try:
                            new_preds = model.predict(new_df)
                            pred_df = pd.DataFrame(new_preds, columns=["Prediction"])
                            result_df = pd.concat([new_df.reset_index(drop=True), pred_df], axis=1)
                            st.success("Prediction completed!")
                            st.dataframe(result_df)
                            st.markdown("---")
                            st.subheader("Download Predictions")
                            st.write("Click the button below to download the predictions as a CSV file.")
                            result_path = "new_data_predictions.csv"
                            result_df.to_csv(result_path, index=False)
                            with open(result_path, "rb") as f:
                                st.download_button("Download Predictions CSV", f, file_name="new_data_predictions.csv", mime="text/csv")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

        elif mode == "Automatic":
            auto_runner = AutoModelSelector()

            if st.button("Run AutoModelSelector with Optuna"):
                if problem_type == "Regression":
                    best_model_name, best_model, best_score = auto_runner.run_regression(X_train, y_train)
                else:
                    best_model_name, best_model, best_score = auto_runner.run_classification(X_train, y_train)

                st.write(f" Best Model: `{best_model_name}`")
                st.write(f" Best Score: `{best_score:.4f}`")

                st.subheader("All Model Scores")
                for name, score in auto_runner.all_models_scores:
                    st.write(f"**{name}**: {score}")

                st.markdown("---")
                st.subheader("Predict on New Data")

                new_data_file = st.file_uploader("Upload new dataset (no target column)", key="auto_new_data")

                if new_data_file is not None:
                    new_df = pd.read_csv(new_data_file)
                    st.dataframe(new_df.head())

                    if st.button("Predict on New Data (Auto Model)"):
                        try:
                            new_preds = best_model.predict(new_df)
                            pred_df = pd.DataFrame(new_preds, columns=["Prediction"])
                            result_df = pd.concat([new_df.reset_index(drop=True), pred_df], axis=1)
                            st.success(" Prediction completed!")
                            st.dataframe(result_df)

                            result_path = "auto_new_data_predictions.csv"
                            result_df.to_csv(result_path, index=False)
                            with open(result_path, "rb") as f:
                                st.download_button("Download Predictions CSV", f, file_name="auto_new_data_predictions.csv", mime="csv")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
