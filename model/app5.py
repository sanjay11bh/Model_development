from model6 import Models, optuna_Model, ReadingData, AutoModelSelector
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import optuna.visualization.matplotlib as optuna_vis

# --- Helper Functions ---

def show_data_info(df):
    st.subheader("Data Summary")
    st.write(df.describe())

def scale_features(X_train, X_test, scaler_name):
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        return X_train, X_test  # No scaling

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)

def display_regression_metrics(r2, mae, mse, rmse):
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

def display_classification_metrics(acc, f1):
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

# --- Main App ---

st.title("AutoML App with Optuna and Streamlit")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    reader = ReadingData()
    df = reader.csv_file(uploaded_file)
    st.dataframe(df.head())

    show_data_info(df)

    all_columns = list(df.columns)
    target_columns = st.multiselect("Select one or more target columns", all_columns)

    if "target_columns" not in st.session_state:
        st.session_state.target_columns = []

    if st.button("Set target"):
        if not target_columns:
            st.error("Please select at least one target column.")
        else:
            st.session_state.target_columns = target_columns
            st.success(f"Target columns set to: {target_columns}")

    if st.session_state.target_columns:
        problem_type = st.radio("Select problem type", ["Regression", "Classification"])

        test_size = st.slider("Select test set size (fraction)", 0.1, 0.5, 0.2, 0.05)

        X = df.drop(columns=st.session_state.target_columns)
        y = df[st.session_state.target_columns]

        # Feature Selection Option
        feature_selection = st.checkbox("Select specific features to use (optional)")
        if feature_selection:
            selected_features = st.multiselect("Select features", X.columns)
            if selected_features:
                X = X[selected_features]
            else:
                st.warning("No features selected, using all features.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Feature Scaling Option
        scaler_option = st.selectbox("Select feature scaling method", ["None", "StandardScaler", "MinMaxScaler"])
        if scaler_option != "None":
            X_train, X_test = scale_features(X_train, X_test, scaler_option)

        mode = st.radio("Select mode", ["Manual", "Automatic"])
        use_optuna = st.checkbox("Use Optuna for hyperparameter tuning")
        use_cv = st.checkbox("Use Cross-Validation (only in Manual mode)")

        if mode == "Manual":
            model_type = st.selectbox("Select model", [
                "Linear Regression", "Lasso Regression", "SVR",
                "Logistic Regression", "Decision Tree", "Random Forest",
                "Gradient Boosting", "AdaBoost", "SVM", "KNN"
            ])

            if st.button("Run Model"):
                if use_optuna:
                    opt_model = optuna_Model()
                    if problem_type == "Regression":
                        if model_type == "Linear Regression":
                            model, pred, r2, mae, mse, rmse = opt_model.linear_regressor(X_train, y_train)
                        elif model_type == "Lasso Regression":
                            model, pred, r2, mae, mse, rmse = opt_model.lasso_regression(X_train, y_train)
                        elif model_type == "SVR":
                            model, pred, r2, mae, mse, rmse = opt_model.svr_regressor(X_train, y_train)
                        elif model_type == "Decision Tree":
                            model, pred, r2, mae, mse, rmse = opt_model.decision_tree_regression(X_train, y_train)
                        elif model_type == "Random Forest":
                            model, pred, r2, mae, mse, rmse = opt_model.random_forest_regression(X_train, y_train)
                        elif model_type == "Gradient Boosting":
                            model, pred, r2, mae, mse, rmse = opt_model.gradient_boosting_regressor(X_train, y_train)
                        elif model_type == "AdaBoost":
                            model, pred, r2, mae, mse, rmse = opt_model.adaboost_regressor(X_train, y_train)
                        else:
                            st.error("Selected regression model not implemented for Optuna yet.")
                            st.stop()

                        display_regression_metrics(r2, mae, mse, rmse)

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
                        elif model_type == "SVM":
                            model, pred, acc, f1 = opt_model.svm_classifier(X_train, y_train)
                        elif model_type == "KNN":
                            model, pred, acc, f1 = opt_model.knn_classifier(X_train, y_train)
                        else:
                            st.error("Selected classification model not implemented for Optuna yet.")
                            st.stop()

                        display_classification_metrics(acc, f1)

                    pred_df, file_path = opt_model.predict_and_save(model, X_test)
                    st.dataframe(pred_df)
                    with open(file_path, "rb") as f:
                        st.download_button("Download Predictions CSV", f, file_name="predictions.csv", mime="text/csv")

                    # Optuna plot
                    if hasattr(opt_model, 'study') and opt_model.study is not None:
                        st.subheader("Optuna Optimization History")
                        fig = optuna_vis.plot_optimization_history(opt_model.study)
                        st.pyplot(fig)
                        plt.clf()

                else:
                    model_runner = Models()

                    if problem_type == "Regression":
                        if model_type == "Linear Regression":
                            model, pred, r2, mae, mse, rmse = model_runner.Linear_regressor(X_train, y_train, cv=use_cv)
                        elif model_type == "Lasso Regression":
                            model, pred, r2, mae, mse, rmse = model_runner.Lasso_regression(X_train, y_train, cv=use_cv)
                        elif model_type == "SVR":
                            model, pred, r2, mae, mse, rmse = model_runner.SVR_regressor(X_train, y_train, cv=use_cv)
                        elif model_type == "Decision Tree":
                            model, pred, r2, mae, mse, rmse = model_runner.Decision_tree_regression(X_train, y_train, cv=use_cv)
                        elif model_type == "Random Forest":
                            model, pred, r2, mae, mse, rmse = model_runner.Random_forest_regression(X_train, y_train, cv=use_cv)
                        elif model_type == "Gradient Boosting":
                            model, pred, r2, mae, mse, rmse = model_runner.Gradient_boosting_regressor(X_train, y_train, cv=use_cv)
                        elif model_type == "AdaBoost":
                            model, pred, r2, mae, mse, rmse = model_runner.AdaBoost_regressor(X_train, y_train, cv=use_cv)
                        else:
                            st.error("Selected regression model not implemented.")
                            st.stop()

                        display_regression_metrics(r2, mae, mse, rmse)

                    else:
                        if model_type == "Logistic Regression":
                            model, pred, acc, f1 = model_runner.Logistic_regression(X_train, y_train, cv=use_cv)
                        elif model_type == "Decision Tree":
                            model, pred, acc, f1 = model_runner.Decision_tree_classifier(X_train, y_train, cv=use_cv)
                        elif model_type == "Random Forest":
                            model, pred, acc, f1 = model_runner.Random_forest_classifier(X_train, y_train, cv=use_cv)
                        elif model_type == "Gradient Boosting":
                            model, pred, acc, f1 = model_runner.Gradient_boosting_classifier(X_train, y_train, cv=use_cv)
                        elif model_type == "AdaBoost":
                            model, pred, acc, f1 = model_runner.AdaBoost_classifier(X_train, y_train, cv=use_cv)
                        elif model_type == "SVM":
                            model, pred, acc, f1 = model_runner.SVM_classifier(X_train, y_train, cv=use_cv)
                        elif model_type == "KNN":
                            model, pred, acc, f1 = model_runner.KNN_classifier(X_train, y_train, cv=use_cv)
                        else:
                            st.error("Selected classification model not implemented.")
                            st.stop()

                        display_classification_metrics(acc, f1)

                    # Save predictions CSV
                    pred_df = pd.DataFrame(pred, columns=st.session_state.target_columns)
                    st.dataframe(pred_df)
                    csv_file = "predictions.csv"
                    pred_df.to_csv(csv_file, index=False)
                    with open(csv_file, "rb") as f:
                        st.download_button("Download Predictions CSV", f, file_name=csv_file, mime="text/csv")

        else:  # Automatic mode
            if st.button("Run AutoModel Selector"):
                auto_selector = AutoModelSelector()
                auto_selector.set_data(X_train, y_train, X_test, y_test, problem_type)
                best_model_name, model, r2, mae, mse, rmse, acc, f1 = auto_selector.automl_run()

                st.success(f"Best model selected: {best_model_name}")

                if problem_type == "Regression":
                    display_regression_metrics(r2, mae, mse, rmse)
                else:
                    display_classification_metrics(acc, f1)

                pred = model.predict(X_test)
                pred_df = pd.DataFrame(pred, columns=st.session_state.target_columns)
                st.dataframe(pred_df)

                csv_file = "predictions.csv"
                pred_df.to_csv(csv_file, index=False)
                with open(csv_file, "rb") as f:
                    st.download_button("Download Predictions CSV", f, file_name=csv_file, mime="text/csv")

else:
    st.info("Upload a CSV file to start.")

