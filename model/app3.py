import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model3 import ReadingData, Models  # Import the models from the script you created
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

# Function to load dataset
def load_data():
    file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "txt"])
    if file:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            data = pd.read_csv(file)  # Assuming it's a CSV for now
        return data
    return None

# Function to evaluate all models and return the best one
def evaluate_all_models(X, y, task_type, use_optuna=False):
    models = []
    if task_type == "Regression":
        models = ["Linear Regression", "Lasso Regression", 
                  "Decision Tree Regressor", "Random Forest Regressor",
                  "Gradient Boosting Regressor"]
    elif task_type == "Classification":
        models = ["Logistic Regression", "Decision Tree Classifier", 
                  "Random Forest Classifier", "Gradient Boosting Classifier"]

    model = Models()  # Initialize the Models class
    best_model = None
    best_metric = -float('inf')  # For regression, we use R2; for classification, we use accuracy

    results = {}
    tuning_results = {}  # Store results for hyperparameter tuning

    for model_choice in models:
        st.write(f"Evaluating {model_choice}...")

        if model_choice == "Linear Regression":
            model_obj, predictions, r2, mae, mse, rmse = model.linear_regressor(X, y)
            results[model_choice] = r2
            if r2 > best_metric:
                best_metric = r2
                best_model = model_obj

        elif model_choice == "Logistic Regression":
            model_obj, predictions, acc, f1 = model.logistic_regression(X, y)
            results[model_choice] = acc
            if acc > best_metric:
                best_metric = acc
                best_model = model_obj

        elif model_choice == "Lasso Regression":
            model_obj, predictions, r2, mae, mse, rmse = model.lasso_regression(X, y)
            results[model_choice] = r2
            if r2 > best_metric:
                best_metric = r2
                best_model = model_obj

        elif model_choice == "Decision Tree Regressor":
            model_obj, predictions, r2, mae, mse, rmse = model.decision_tree_regression(X, y, use_optuna)
            results[model_choice] = r2
            if r2 > best_metric:
                best_metric = r2
                best_model = model_obj

        elif model_choice == "Decision Tree Classifier":
            model_obj, predictions, acc, f1 = model.decision_tree_classifier(X, y, use_optuna)
            results[model_choice] = acc
            if acc > best_metric:
                best_metric = acc
                best_model = model_obj

        elif model_choice == "Random Forest Regressor":
            model_obj, predictions, r2, mae, mse, rmse = model.random_forest_regression(X, y, use_optuna)
            results[model_choice] = r2
            if r2 > best_metric:
                best_metric = r2
                best_model = model_obj

        elif model_choice == "Random Forest Classifier":
            model_obj, predictions, acc, f1 = model.random_forest_classifier(X, y, use_optuna)
            results[model_choice] = acc
            if acc > best_metric:
                best_metric = acc
                best_model = model_obj

        elif model_choice == "Gradient Boosting Regressor":
            model_obj, predictions, r2, mae, mse, rmse = model.gradient_boosting_regressor(X, y, use_optuna)
            results[model_choice] = r2
            if r2 > best_metric:
                best_metric = r2
                best_model = model_obj

        elif model_choice == "Gradient Boosting Classifier":
            model_obj, predictions, acc, f1 = model.gradient_boosting_classifier(X, y, use_optuna)
            results[model_choice] = acc
            if acc > best_metric:
                best_metric = acc
                best_model = model_obj

        if use_optuna:
            study = optuna.create_study(direction="maximize" if task_type == "Classification" else "minimize")
            study.optimize(lambda trial: model(model_choice, X, y, trial, task_type), n_trials=10)
            tuning_results[model_choice] = study.best_value

    return best_model, results, tuning_results

# Function to plot the graphs
def plot_performance_comparison(results, tuning_results):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the model performance (Accuracy/R2)
    ax[0].barh(list(results.keys()), list(results.values()), color='skyblue')
    ax[0].set_xlabel("Performance Metric")
    ax[0].set_title("Model Performance Comparison")

    # Plot the hyperparameter tuning results (Optuna)
    ax[1].barh(list(tuning_results.keys()), list(tuning_results.values()), color='salmon')
    ax[1].set_xlabel("Optuna Best Value")
    ax[1].set_title("Hyperparameter Tuning Results")

    plt.tight_layout()
    st.pyplot(fig)

# Function to download results as CSV
def download_results(results, filename="model_results.csv"):
    results_df = pd.DataFrame(results.items(), columns=["Model", "Score"])
    results_csv = results_df.to_csv(index=False)
    st.download_button(label="Download Results as CSV", data=results_csv, file_name=filename, mime="text/csv")

# Main function to run the app
def main():
    st.title("Automated Model Selection and Hyperparameter Tuning with Optuna")

    # Load the dataset
    data = load_data()

    if data is not None:
        st.write("Dataset loaded successfully!")
        st.write(data.head())  # Show the first few rows of the dataset

        # Dataset Preview Button
        if st.button("Preview Dataset"):
            st.write(data)

        # Allow the user to choose one or more target columns
        target_columns = st.multiselect("Select target column(s)", data.columns)
        if target_columns:
            st.write(f"Target Columns: {target_columns}")

            # Split the data into features and targets
            X = data.drop(target_columns, axis=1)
            y = data[target_columns]

            # Standardize the data (Optional, but recommended for many models)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Choose task type: Classification or Regression
            task_type = st.radio("Select task type", ("Classification", "Regression"))

            # Choose to evaluate all models
            evaluate_all = st.checkbox("Evaluate all models and find the best one")

            # Evaluate models and display best model
            if evaluate_all:
                if st.button("Evaluate Models"):
                    # Evaluate all models and get the best one
                    best_model, results, tuning_results = evaluate_all_models(X_scaled, y, task_type, use_optuna=True)
                    st.write(f"Best model is: {best_model}")
                    st.write("Model performance across all models:")
                    st.write(results)

                    # Plot the performance comparison and tuning results
                    plot_performance_comparison(results, tuning_results)

                    # Allow the user to download results as CSV
                    download_results(results)

            else:
                # Allow the user to choose a specific model based on task type
                if task_type == "Regression":
                    model_choice = st.selectbox(
                        "Select the regression model for training",
                        ("Linear Regression", "Lasso Regression", 
                         "Decision Tree Regressor", "Random Forest Regressor",
                         "Gradient Boosting Regressor")
                    )
                elif task_type == "Classification":
                    model_choice = st.selectbox(
                        "Select the classification model for training",
                        ("Logistic Regression", "Decision Tree Classifier", 
                         "Random Forest Classifier", "Gradient Boosting Classifier")
                    )

                # Whether to use Optuna for hyperparameter tuning
                use_optuna = st.checkbox("Use Optuna for hyperparameter tuning")

                # Initialize the model class
                model = Models()

                # Train the selected model
                if st.button("Train Model"):
                    if model_choice == "Linear Regression":
                        model_obj, predictions, r2, mae, mse, rmse = model.linear_regressor(X_scaled, y)
                        st.write(f"R2 Score: {r2}")
                        st.write(f"MAE: {mae}")
                        st.write(f"MSE: {mse}")
                        st.write(f"RMSE: {rmse}")

                    elif model_choice == "Logistic Regression":
                        model_obj, predictions, acc, f1 = model.logistic_regression(X_scaled, y)
                        st.write(f"Accuracy: {acc}")
                        st.write(f"F1 Score: {f1}")

                    elif model_choice == "Lasso Regression":
                        model_obj, predictions, r2, mae, mse, rmse = model.lasso_regression(X_scaled, y)
                        st.write(f"R2 Score: {r2}")
                        st.write(f"MAE: {mae}")
                        st.write(f"MSE: {mse}")
                        st.write(f"RMSE: {rmse}")

                    elif model_choice == "Decision Tree Regressor":
                        model_obj, predictions, r2, mae, mse, rmse = model.decision_tree_regression(X_scaled, y, use_optuna)
                        st.write(f"R2 Score: {r2}")
                        st.write(f"MAE: {mae}")
                        st.write(f"MSE: {mse}")
                        st.write(f"RMSE: {rmse}")

                    elif model_choice == "Decision Tree Classifier":
                        model_obj, predictions, acc, f1 = model.decision_tree_classifier(X_scaled, y, use_optuna)
                        st.write(f"Accuracy: {acc}")
                        st.write(f"F1 Score: {f1}")

                    elif model_choice == "Random Forest Regressor":
                        model_obj, predictions, r2, mae, mse, rmse = model.random_forest_regression(X_scaled, y, use_optuna)
                        st.write(f"R2 Score: {r2}")
                        st.write(f"MAE: {mae}")
                        st.write(f"MSE: {mse}")
                        st.write(f"RMSE: {rmse}")

                    elif model_choice == "Random Forest Classifier":
                        model_obj, predictions, acc, f1 = model.random_forest_classifier(X_scaled, y, use_optuna)
                        st.write(f"Accuracy: {acc}")
                        st.write(f"F1 Score: {f1}")

                    elif model_choice == "Gradient Boosting Regressor":
                        model_obj, predictions, r2, mae, mse, rmse = model.gradient_boosting_regressor(X_scaled, y, use_optuna)
                        st.write(f"R2 Score: {r2}")
                        st.write(f"MAE: {mae}")
                        st.write(f"MSE: {mse}")
                        st.write(f"RMSE: {rmse}")

                    elif model_choice == "Gradient Boosting Classifier":
                        model_obj, predictions, acc, f1 = model.gradient_boosting_classifier(X_scaled, y, use_optuna)
                        st.write(f"Accuracy: {acc}")
                        st.write(f"F1 Score: {f1}")

if __name__ == "__main__":
    main()
