import streamlit as st
import pandas as pd
from model6 import ReadingData, Models, optuna_Model, AutoModelSelector
from sklearn.model_selection import train_test_split

def main():
    st.title("Multi-Model ML Pipeline")

    rd = ReadingData()
    manual = Models()
    optuna = optuna_Model()
    auto = AutoModelSelector()

    st.markdown("### Upload your dataset")
    uploaded_file = st.file_uploader("Upload CSV, XLSX or TXT file", type=["csv", "xlsx", "txt"])

    if uploaded_file:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        data = rd.read_data(file_path)
        st.success("Data Loaded Successfully!")
        st.dataframe(data.head())
        
        all_columns = list(data.columns)
        target_columns = st.multiselect("Select one or more target columns", all_columns)

        if "target_columns" not in st.session_state:
            st.session_state.target_columns = []

        if st.button("Set target"):
            st.session_state.target_columns = target_columns
            st.success(f"Target columns set to: {target_columns}")

        test_size = st.slider("Select test size fraction", 0.0, 0.5, 0.3, 0.05)

        if st.session_state.target_columns:
            X = data.drop(columns=st.session_state.target_columns)
            X.columns = X.columns.astype(str)
            y = data[st.session_state.target_columns]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


        model_flow = st.radio("Select model run type", ("Manual", "Optuna Tuning", "Automated"))

        task_type = st.radio("Select task type", ("Regression", "Classification"))

        if model_flow == "Manual":
            if task_type == "Regression":
                    model_choice = st.selectbox("Select regression model", (
                        "Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet Regression",
                        "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost",
                        "SVR", "XGBoost Regressor"
                    ))

            else:
                model_choice = st.selectbox("Select classification model", (
                    "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost",
                    "SVR Classifier", "KNN Classifier", "XGBoost Classifier"
                ))

            if st.button("Run Model"):
                if task_type == "Regression":
                    if model_choice == "Linear Regression":
                        model, predictions, r2, mae, mse, rmse = manual.Linear_regressor(X_train, X_test, y_train, y_test)
                    elif model_choice == "Lasso Regression":
                        model, predictions, r2, mae, mse, rmse = manual.Lasso_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Ridge Regression":
                        model, predictions, r2, mae, mse, rmse = manual.Ridge_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "ElasticNet Regression":
                        model, predictions, r2, mae, mse, rmse = manual.ElasticNet_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Decision Tree":
                        model, predictions, r2, mae, mse, rmse = manual.Decision_tree_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, predictions, r2, mae, mse, rmse = manual.Random_forest_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Gradient Boosting":
                        model, predictions, r2, mae, mse, rmse = manual.Gradient_boosting_regressor(X_train, X_test, y_train, y_test)
                    elif model_choice == "AdaBoost":
                        model, predictions, r2, mae, mse, rmse = manual.AdaBoost_regressor(X_train, X_test, y_train, y_test)
                    elif model_choice == "SVR":
                        model, predictions, r2, mae, mse, rmse = manual.SVR_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "XGBoost Regressor":
                        model, predictions, r2, mae, mse, rmse = manual.Xgb_regressor(X_train, X_test, y_train, y_test)

                    manual.save_predictions_to_csv(y_test, predictions)
                    manual.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)
                    manual.k_fold_cross_validation(model, X, y, n_splits=5, is_classification=False)

                    st.write(f"R²: {r2:.4f}")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"MSE: {mse:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")

                else:  # classification
                    if model_choice == "Logistic Regression":
                        model, pred, acc, f1 = manual.Logistic_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Decision Tree":
                        model, pred, acc, f1 = manual.Decision_tree_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, pred, acc, f1 = manual.Random_forest_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "Gradient Boosting":
                        model, pred, acc, f1 = manual.Gradient_boosting_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "AdaBoost":
                        model, pred, acc, f1 = manual.AdaBoost_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "SVR Classifier":
                        model, pred, acc, f1 = manual.SVR_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "KNN Classifier":
                        model, pred, acc, f1 = manual.KNN_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "XGBoost Classifier":
                        model, pred, acc, f1 = manual.XGBoost_classifier(X_train, X_test, y_train, y_test)

                    manual.save_predictions_to_csv(y_test, pred)
                    manual.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)
                    manual.k_fold_cross_validation(model, X, y, n_splits=5, is_classification=True)

                    st.write(f"Accuracy: {acc:.4f}")
                    st.write(f"F1 Score: {f1:.4f}")


        elif model_flow == "Optuna Tuning":
            if task_type == "Regression":
                model_choice = st.selectbox("Select regression model", (
                    "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost", "SVR", "XGBoost Regressor"
                ))

                if st.button("Run Optuna Regression"):
                    if model_choice == "Decision Tree":
                        model, predictions, r2, mae, mse, rmse = optuna.Decision_tree_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, predictions, r2, mae, mse, rmse = optuna.Random_forest_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Gradient Boosting":
                        model, predictions, r2, mae, mse, rmse = optuna.Gradient_boosting_regressor(X_train, X_test, y_train, y_test)
                    elif model_choice == "AdaBoost":
                        model, predictions, r2, mae, mse, rmse = optuna.AdaBoost_regressor(X_train, X_test, y_train, y_test)
                    elif model_choice == "SVR":
                        model, predictions, r2, mae, mse, rmse = optuna.SVR_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "XGBoost Regressor":
                        model, predictions, r2, mae, mse, rmse = optuna.Xgb_regressor(X_train, X_test, y_train, y_test)

                    optuna.save_predictions_to_csv(y_test, predictions)
                    optuna.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)
                    optuna.optuna_visualization()

                    st.write(f"R²: {r2:.4f}")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"MSE: {mse:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")

            else:
                model_choice = st.selectbox("Select classification model", (
                    "Decision Tree", "Random Forest", "Gradient Boosting", "AdaBoost",
                    "SVR Classifier", "KNN Classifier", "XGBoost Classifier"
                ))

                if st.button("Run Optuna Classification"):
                    if model_choice == "Decision Tree":
                        model, pred, acc, f1 = optuna.Decision_tree_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, pred, acc, f1 = optuna.Random_forest_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "Gradient Boosting":
                        model, pred, acc, f1 = optuna.Gradient_boosting_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "AdaBoost":
                        model, pred, acc, f1 = optuna.AdaBoost_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "SVR Classifier":
                        model, pred, acc, f1 = optuna.SVR_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "KNN Classifier":
                        model, pred, acc, f1 = optuna.KNN_classifier(X_train, X_test, y_train, y_test)
                    elif model_choice == "XGBoost Classifier":
                        model, pred, acc, f1 = optuna.XGBoost_classifier(X_train, X_test, y_train, y_test)

                    optuna.save_predictions_to_csv(y_test, pred)
                    optuna.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)
                    ## optuna.k_fold_cross_validation(model, X, y, n_splits=5, is_classification=True)

                    st.write(f"Accuracy: {acc:.4f}")
                    st.write(f"F1 Score: {f1:.4f}")


        else:
            st.write("Automated Model Selection")
            if st.button("Run Auto Model"):
                if task_type == "Regression":
                    auto.run_regression(X_train, X_test, y_train, y_test)
                    st.success("Automated Regression Model Run Complete")
                else:
                    auto.run_classification(X_train, X_test, y_train, y_test)
                    st.success("Automated Classification Model Run Complete")

if __name__ == "__main__":
    main()
