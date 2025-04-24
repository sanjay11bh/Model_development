import streamlit as st
from final3 import SpectralProcessor  # Refactored version of your script
from model2 import ReadingData, Models, ModelSelector
import pandas as pd

st.set_page_config(layout="wide")
st.title("Spectral Preprocessing + Model Selection App")

# ========== Sidebar Input ==========
st.sidebar.header("Upload Spectral Data File")
spectral_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if spectral_file:
    st.sidebar.write("Preprocessing Configuration")
    spectral_type = st.sidebar.selectbox("Spectral Type", ["IR - Transmission", "IR - Absorbance", "Raman", "UV-Vis"])
    spectral_type_map = {
        "IR - Transmission": '1',
        "IR - Absorbance": '2',
        "Raman": '3',
        "UV-Vis": '4'
    }
    type_choice = spectral_type_map[spectral_type]

    processor = SpectralProcessor(choice=type_choice, file=spectral_file)

    st.write("### Raw Data Preview")
    st.dataframe(processor.data.spc.head())

    # --- Add preprocessing steps here using Streamlit widgets ---
    st.sidebar.write("### Preprocessing Steps")
    if st.sidebar.button("Apply Normalization"):
        processor.data.snv()

    if st.sidebar.button("Apply Preprocessing and Plot"):
        processor.plot_2d_data()  # or 3D if you prefer

    st.sidebar.write("### Ready for ML?")
    if st.sidebar.button("Proceed to Modeling"):
        processed_df = processor.get_processed_data()
        df = pd.DataFrame(processed_df)

        st.write("### Processed Data for ML")
        st.dataframe(df.head())

        # ========== Model Development ==========

        all_columns = list(df.columns)
        target_column = st.selectbox("Select Target Column", all_columns)

        if target_column:
            task_type = st.radio("Select Task Type", ["Regression", "Classification"], horizontal=True)

            df = df.dropna()
            X = df.drop(columns=[target_column])
            y = df[target_column]

            if y.dtypes == 'object':
                y = pd.factorize(y)[0]

            modeler = Models()

            if task_type == "Regression":
                model = modeler.linear_regressor(X, y)
                st.success("Regression model trained!")
                st.write(model)
            else:
                model = modeler.random_forest_classifier(X, y)
                st.success("Classification model trained!")
                st.write(model)
