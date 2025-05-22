import streamlit as st
import pandas as pd
import plotly.express as px
from model2 import ReadingData, Models

st.set_page_config(page_title="ML model ", layout="wide")
st.title("Model Selection App")

# Initialize helper classes
reader = ReadingData()
modeler = Models()

st.sidebar.header("1. Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == "csv":
        df = reader.csv_file(uploaded_file)
    else:
        df = reader.xlsx_file(uploaded_file)

    st.subheader(" Preview of the Dataset")
    st.dataframe(df.head(), use_container_width=True)

    with st.expander(" Dataset Information"):
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.write("Null values per column:")
        st.dataframe(df.isnull().sum())

    all_columns = list(df.columns)
    target_column = st.selectbox(" Select the target column", all_columns)

    if target_column:
        task_type = st.radio(" Select Task Type", ["Regression", "Classification"], horizontal=True)

        df = df.dropna()
        X = df.drop(columns=[target_column])
        y = df[target_column]

        if y.dtypes == 'object':
            y = pd.factorize(y)[0]
        # Model dictionary  

        if task_type == "Regression":
            models_dict = {
                "Linear Regression": modeler.linear_regressor,
                "Decision Tree": modeler.decision_tree_regression,
                "Lasso Regression": modeler.lasso_regression,
                "Random Forest": modeler.random_forest_regression,
                "Gradient Boosting Regressor": modeler.gradient_boosting_regressor,
            }
        else:
            models_dict = {
                "Logistic Regression": modeler.logistic_regression,
                "Decision Tree": modeler.decision_tree_classifier,
                "Random Forest": modeler.random_forest_classifier,
                "Gradient Boosting": modeler.gradient_boosting_classifier
            }
        # Model selection
        model_name = st.selectbox(" Select the model", list(models_dict.keys()))
        model = models_dict[model_name]
        model.fit(X, y) 
        y_pred = model.predict(X)
        st.subheader("Model Performance")
        if task_type == "Regression":
            st.write("Mean Absolute Error:", modeler.mean_absolute_error(y, y_pred))
            st.write("Mean Squared Error:", modeler.mean_squared_error(y, y_pred))
            st.write("Root Mean Squared Error:", modeler.root_mean_squared_error(y, y_pred))
            st.write("R2 Score:", modeler.r2_score(y, y_pred))
        else:
            st.write("Accuracy:", modeler.accuracy_score(y, y_pred))
            st.write("F1 Score:", modeler.f1_score(y, y_pred))
            st.write("Confusion Matrix:")
            fig = px.imshow(modeler.confusion_matrix(y, y_pred), text_auto=True)
            st.plotly_chart(fig)