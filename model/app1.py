import streamlit as st
import pandas as pd
import plotly.express as px
from model1 import ReadingData, Models, ModelSelector

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
    
                "AdaBoost Regressor": modeler.adaboost_regressor,
                "Bagging Regressor": modeler.bagging_regressor,
                "Gradient Boosting Regressor": modeler.gradient_boosting_regressor,
            }
        else:
            models_dict = {
                "Logistic Regression": modeler.logistic_regression,
                "Decision Tree": modeler.decision_tree_classifier,
                "Random Forest": modeler.random_forest_classifier,
                "SGD Classifier": modeler.SGD_classifier,
                "AdaBoost": modeler.adaboost_classifier,
                "Bagging Classifier": modeler.bagging_classifier,
                "Gradient Boosting": modeler.gradient_boosting_classifier
            }

        # Selection Mode
        st.sidebar.header("2. Train & Evaluate Models")
        selection_mode = st.radio("Model Selection Mode", ["Automatic Best Model Selection", "Choose a model manually"])

        if selection_mode == "Choose a model manually":
            selected_model_name = st.selectbox("Select a model to train", list(models_dict.keys()))

        if st.sidebar.button("Start Training"):
            selector = ModelSelector(task='regression' if task_type == 'Regression' else 'classification')

            with st.spinner(" Training in progress..."):

                if selection_mode == "Automatic Best Model Selection":
                    for name, model_func in models_dict.items():
                        try:
                            model, _, *metrics = model_func(X, y)
                            selector.results[name] = metrics[0]
                        except Exception as e:
                            st.warning(f"{name} failed: {e}")

                    best_model_name, best_score = selector.run(models_dict, X, y)

                    st.success(f" Best Model: `{best_model_name}`")
                    st.metric(label="Best Score", value=f"{best_score:.4f}")

                    # Plot all model scores
                    st.subheader(" Model Performance Comparison")
                    score_series = pd.Series(selector.results).sort_values(ascending=False)
                    fig = px.bar(score_series, x=score_series.index, y=score_series.values,
                                 labels={'x': 'Model', 'y': 'Score'}, title='Model Performance')
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    try:
                        model, _, *metrics = models_dict[selected_model_name](X, y)
                        score = metrics[0]
                        st.success(f"Trained `{selected_model_name}` successfully!")
                        st.metric(label="Score", value=f"{score:.4f}")
                    except Exception as e:
                        st.error(f" {selected_model_name} failed to train: {e}")
