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
        # Save uploaded file temporarily for ReadingData module to read
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read data
        data = rd.read_data(file_path)
        st.success("Data Loaded Successfully!")
        st.dataframe(data.head())

        # Target selection
        target = st.selectbox("Select target column", options=data.columns)

        # Test size input
        test_size = st.slider("Select test size fraction", 0.0, 0.5, 0.3, 0.05)

        # Split data
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model flow selection
        model_flow = st.radio("Select model run type", ("Manual", "Optuna Tuning", "Automated"))

        # Task selection
        task_type = st.radio("Select task type", ("Regression", "Classification"))

        if model_flow == "Manual":
            if task_type == "Regression":
                model_choice = st.selectbox("Select regression model",
                    ("Linear Regression", "Random Forest", "AdaBoost"))
            else:
                model_choice = st.selectbox("Select classification model",
                    ("Logistic Regression", "Random Forest", "AdaBoost"))

            if st.button("Run Model"):
                if task_type == "Regression":
                    if model_choice == "Linear Regression":
                        model, predictions, r2, mae, mse, rmse = manual.Linear_regressor(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, predictions, r2, mae, mse, rmse = manual.Random_forest_regression(X_train, X_test, y_train, y_test)
                    else:
                        model, predictions, r2, mae, mse, rmse = manual.AdaBoost_regressor(X_train, X_test, y_train, y_test)

                    manual.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)
                    manual.save_predictions_to_csv(y_test, predictions)
                    manual.k_fold_cross_validation(model, X, y, n_splits=5, is_classification=False)

                    st.write(f"R²: {r2:.4f}")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"MSE: {mse:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")

                else:
                    if model_choice == "Logistic Regression":
                        model, pred, acc, f1 = manual.Logistic_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, pred, acc, f1 = manual.Random_forest_classifier(X_train, X_test, y_train, y_test)
                    else:
                        model, pred, acc, f1 = manual.AdaBoost_classifier(X_train, X_test, y_train, y_test)

                    manual.save_predictions_to_csv(y_test, pred)
                    manual.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)
                    manual.k_fold_cross_validation(model, X, y, n_splits=5, is_classification=True)

                    st.write(f"Accuracy: {acc:.4f}")
                    st.write(f"F1 Score: {f1:.4f}")

        elif model_flow == "Optuna Tuning":
            if task_type == "Regression":
                model_choice = st.selectbox("Select regression model",
                    ("Decision Tree", "Random Forest", "AdaBoost"))

                if st.button("Run Optuna Regression"):
                    if model_choice == "Decision Tree":
                        model, predictions, r2, mae, mse, rmse = optuna.Decision_tree_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, predictions, r2, mae, mse, rmse = optuna.Random_forest_regression(X_train, X_test, y_train, y_test)
                    else:
                        model, predictions, r2, mae, mse, rmse = manual.AdaBoost_regressor(X_train, X_test, y_train, y_test)

                    optuna.save_predictions_to_csv(y_test, predictions)
                    optuna.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=False)
                    st.write(f"R²: {r2:.4f}")
                    st.write(f"MAE: {mae:.4f}")
                    st.write(f"MSE: {mse:.4f}")
                    st.write(f"RMSE: {rmse:.4f}")

            else:
                model_choice = st.selectbox("Select classification model",
                    ("Logistic Regression", "Random Forest", "AdaBoost"))

                if st.button("Run Optuna Classification"):
                    if model_choice == "Logistic Regression":
                        model, pred, acc, f1 = optuna.Logistic_regression(X_train, X_test, y_train, y_test)
                    elif model_choice == "Random Forest":
                        model, pred, acc, f1 = optuna.Random_forest_classifier(X_train, X_test, y_train, y_test)
                    else:
                        model, pred, acc, f1 = optuna.AdaBoost_classifier(X_train, X_test, y_train, y_test)

                    optuna.save_predictions_to_csv(y_test, pred)
                    optuna.fit_predict_evaluate(model, X_train, X_test, y_train, y_test, is_classification=True)
                    st.write(f"Accuracy: {acc:.4f}")
                    st.write(f"F1 Score: {f1:.4f}")

        else:
            st.write("Automated Model Selection")
            if st.button("Run Auto Model"):
                if task_type == "Regression":
                    auto.run_regression(X_train, y_train, X_test, y_test)
                    st.success("Automated Regression Model Run Complete")
                else:
                    auto.run_classification(X_train, y_train, X_test, y_test)
                    st.success("Automated Classification Model Run Complete")

if __name__ == "__main__":
    main()
