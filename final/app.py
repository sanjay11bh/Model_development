import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from data_augmentation import DataAugmentor
from midel import ReadingData , Models , optuna_Model , AutoModelSelector , WaveletDenoiser , OutlierRemover
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import SpectralData
import tempfile
import numpy as np



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

        if "original_data" not in st.session_state:
            data = rd.read_data(file_path)
            st.session_state.original_data = data.copy() 
            st.session_state.data = data.copy()           
            st.success("Data Loaded Successfully!")

        data = st.session_state.data

        drop_column = st.multiselect("Select column(s) to drop (optional)", data.columns)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Drop Selected Columns"):
                if drop_column:
                    data.drop(columns=drop_column, inplace=True)
                    st.session_state.data = data  
                    st.success(f"Dropped column(s): {', '.join(drop_column)}")
                else:
                    st.warning("Please select at least one column to drop.")

        with col2:
            if st.button("🔄 Reset Dataset"):
                st.session_state.data = st.session_state.original_data.copy()
                st.success("Dataset has been reset to original uploaded version.")

        st.dataframe(data.head())

        # ===== Target Selection =====
        all_columns = list(data.columns)
        target_columns = st.multiselect("Select one or more target columns", all_columns)

        if "target_columns" not in st.session_state:
            st.session_state.target_columns = []

        if st.button("Set target"):
            st.session_state.target_columns = target_columns
            st.success(f"Target columns set to: {target_columns}")

    if "target_columns" in st.session_state and st.session_state.target_columns:
        X = data.drop(columns=st.session_state.target_columns)
    else:
        X = data
    try:
        x_axis = X.columns.astype(float)
    except ValueError as e:
        st.error(f"Cannot convert column names to float: {e}")
        x_axis = np.arange(X.shape[1])

    st.markdown("### 📊 Spectra Before Any Preprocessing")
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(X.shape[0]):
        ax.plot(x_axis, X.iloc[i].values, alpha=0.5)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Before Any Preprocessing")
    ax.grid(True)
    st.pyplot(fig)

################################################################################  WAVELET DENOISING ##########################################################################

    st.markdown("## 🌊 Wavelet Denoising: Full Spectra")

    wavelet = st.selectbox("Wavelet Type", ["db4", "sym4", "rbio4.4", "coif1"], index=2)
    level = st.slider("Decomposition Level", 1, 5, 3)
    mode = st.radio("Thresholding Mode", ["soft", "hard"], horizontal=True)

    denoiser = WaveletDenoiser(wavelet=wavelet, level=level, threshold_mode=mode)
    denoised_df_wavelet = denoiser.denoise_dataframe(X)

    st.markdown("### 📉 Denoised Spectra Visualization")
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(denoised_df_wavelet.shape[0]):
        ax.plot(x_axis, denoised_df_wavelet.iloc[i].values, alpha=0.5)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title("Wavelet Denoised Spectra")
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)


##################################################################   Outlier Removal ##########################################################################################

# ====================== Outlier Removal (Optional) ======================
    st.markdown("##  Optional: Outlier Removal")

    apply_outlier_removal = st.checkbox("Apply outlier removal based on standard deviation?", value=False)

    if apply_outlier_removal:
        threshold = st.slider("Outlier Detection Threshold (σ)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)

        if st.button("Remove Outliers"):
            remover = OutlierRemover(threshold=threshold)
            filtered_spectra_df = remover.fit_transform(denoised_df_wavelet)

            st.success(f"Removed {denoised_df_wavelet.shape[0] - filtered_spectra_df.shape[0]} outliers.")
            st.markdown(f"**Remaining Samples:** {filtered_spectra_df.shape[0]}")

            # Plot filtered 
            fig, ax = plt.subplots(figsize=(14, 6))
            for i in range(filtered_spectra_df.shape[0]):
                ax.plot(x_axis, filtered_spectra_df.iloc[i].values, alpha=0.5)
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Intensity")
            ax.set_title("Spectra After Outlier Removal")
            ax.grid(True)
            st.pyplot(fig)

            denoised_df_wavelet = filtered_spectra_df
    else:
        st.info("Outlier removal is skipped. Proceeding with denoised spectra as-is.")

    
################################################################### PREPROCESSING  Method  #############################################################################################


    st.markdown("## 🧪 Preprocessing of Spectral Data")

    den_df = denoised_df_wavelet.copy()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        den_df.to_csv(tmp.name, index=False)
        spectral = SpectralData(tmp.name)

    techniques = st.multiselect(
        "Select preprocessing techniques (apply sequentially):",
        ['Trim', 'Baseline Correction', 'Smoothing', 'Normalization', 'Center', 'Derivative', 'Dataset']
    )

    processed_steps = []

    for technique in techniques:
        st.markdown(f"### 🔧 {technique}")

        if technique == 'Trim':
            trim_type = st.radio("Trim Type:", ["Trim", "Inverse trim"], key=technique+"_type")
            start = st.number_input("Start Wavelength", value=float(spectral.wav.min()), key=technique+"_start")
            end = st.number_input("End Wavelength", value=float(spectral.wav.max()), key=technique+"_end")

            if st.button(f"Apply {technique}", key=technique+"_btn"):
                if trim_type == "Trim":
                    spectral.trim(start=start, end=end)
                else:
                    spectral.invtrim(start=start, end=end)
                processed_steps.append(f"{technique} ({trim_type})")

        elif technique == "Baseline Correction":
            methods = st.multiselect("Select methods:", ['AsLS', 'Polyfit', 'Pearson'], key="baseline_methods")
            for method in methods:
                if method == "AsLS":
                    lam = st.number_input("λ (Smoothness)", value=1e5, key="asls_lam")
                    p = st.number_input("p (Asymmetry)", value=0.001, key="asls_p")
                    niter = st.number_input("Iterations", value=10, step=1, key="asls_niter")
                    spectral.AsLS(lam=lam, p=p, niter=int(niter))
                elif method == "Polyfit":
                    order = st.number_input("Polynomial Order", value=3, key="poly_order")
                    niter = st.number_input("Iterations", value=1, key="poly_niter")
                    spectral.polyfit(order=int(order), niter=int(niter))
                elif method == "Pearson":
                    u = st.number_input("u (step)", value=4, key="pearson_u")
                    v = st.number_input("v (degree)", value=3, key="pearson_v")
                    spectral.pearson(u=int(u), v=int(v))
                processed_steps.append(f"Baseline - {method}")

        elif technique == "Smoothing":
            methods = st.multiselect("Smoothing method(s):", ['Rolling', 'Savitzky-Golay'], key="smooth")
            for method in methods:
                if method == "Rolling":
                    window = st.number_input("Rolling Window", value=3, step=1, key="roll_win")
                    spectral.rolling(window=int(window))
                elif method == "Savitzky-Golay":
                    window = st.number_input("SG Window", value=5, step=2, key="sg_win")
                    poly = st.number_input("SG Polyorder", value=2, step=1, key="sg_poly")
                    spectral.SGSmooth(window=int(window), poly=int(poly))
                processed_steps.append(f"Smoothing - {method}")

        elif technique == "Normalization":
            norm_methods = st.multiselect("Normalization method(s):", [
                'SNV', 'Detrend', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max', 'Pareto'
            ], key="norm")
            for method in norm_methods:
                if method == "SNV": spectral.snv()
                elif method == "MSC": spectral.msc()
                elif method == "Detrend":
                    order = st.number_input("Detrend Order", value=2, key="detrend_order")
                    spectral.detrend(order=order)
                elif method == "Area": spectral.area()
                elif method == "Peak Normalization":
                    wave = st.number_input("Peak Wavenumber", value=float(spectral.wav.mean()), key="peaknorm")
                    spectral.peaknorm(wavenumber=wave)
                elif method == "Vector": spectral.vector()
                elif method == "Min-max":
                    minv = st.number_input("Min", value=0.0, key="minval")
                    maxv = st.number_input("Max", value=1.0, key="maxval")
                    spectral.minmax(min_val=minv, max_val=maxv)
                elif method == "Pareto": spectral.pareto()
                processed_steps.append(f"Normalization - {method}")

        elif technique == 'Center':
            center_methods = st.multiselect("Centering method(s):", ['Mean (spectrum)', 'Mean (wavelength)', 'Last Point'], key="center")
            for method in center_methods:
                if method == 'Mean (spectrum)': spectral.mean_center(option=False)
                elif method == 'Mean (wavelength)': spectral.mean_center(option=True)
                elif method == 'Last Point': spectral.lastpoint()
                processed_steps.append(f"Center - {method}")

        elif technique == 'Derivative':
            derivative_options = st.multiselect("Derivative Options:", ['Subtract', 'Reset'], key="deriv")
            for option in derivative_options:
                if option == "Subtract":
                    index = st.number_input("Spectrum Index to Subtract (1-based)", min_value=1, value=1, key="subtract_idx")
                    spectral.subtract(spectra=index)
                elif option == "Reset":
                    spectral.reset()
                processed_steps.append(f"Derivative - {option}")

        elif technique == 'Dataset':
            st.markdown("#### SG Derivative")
            window = st.number_input("Window", value=5, step=2, key="ds_window")
            poly = st.number_input("Polyorder", value=2, key="ds_poly")
            order = st.number_input("Order", value=1, key="ds_order")
            spectral.SGDeriv(window=int(window), poly=int(poly), order=int(order))
            processed_steps.append("Dataset - SG Derivative")

        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(len(spectral.spc)):
            ax.plot(spectral.wav, spectral.spc.iloc[i])
        ax.set_title(f"After {technique}")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        st.pyplot(fig)

    if processed_steps:
        st.markdown("###  Applied Preprocessing Steps:")
        for step in processed_steps:
            st.write(f"- {step}")

    if st.button(" Save Preprocessed Spectral Data"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_out:
            df_out = pd.DataFrame(spectral.spc).T
            df_out.insert(0, "Wavelength", spectral.wav)
            tmp_out_path = tmp_out.name
            df_out.to_csv(tmp_out_path, index=False)
            with open(tmp_out_path, "rb") as f:
                st.download_button("Download Processed Data", data=f, file_name="processed_spectra.csv", mime="text/csv")
    if processed_steps:
        st.session_state['preprocessed_X'] = spectral.spc.copy()

        # Make sure you grab correct `y` from original data
        if 'target_columns' in st.session_state:
            target_df = data[st.session_state['target_columns']]
            st.session_state['preprocessed_y'] = target_df.reset_index(drop=True)


###################################################################################  DATA Augmentation  ############################################################################

    st.markdown("## 🔁 Iterative Data Augmentation")

    # Step 1: Load preprocessed data or previous augmented
    X_current = st.session_state.get('augmented_X', st.session_state['preprocessed_X'].copy())
    y_current = st.session_state.get('augmented_y', st.session_state['preprocessed_y'].copy())

    # Track history
    if 'augmentation_history' not in st.session_state:
        st.session_state['augmentation_history'] = []

    # Step 2: Select one technique at a time
    method = st.selectbox("👉 Select one augmentation technique:", [
        'None', 'Add Spectra', 'Mixup', 'Spectral Shift', 'Gaussian Noise'
    ])

    # Step 3: Show parameters specific to the selected method
    params = {}
    apply = False

    if method == 'Add Spectra':
        st.markdown("**🧪 Parameters for Add Spectra**")
        apply = st.button("Apply Add Spectra")
    elif method == 'Mixup':
        st.markdown("**🧪 Parameters for Mixup**")
        params['num_copies'] = st.number_input("Number of synthetic samples", 1, 100, 2, 1, key="mixup_copies")
        params['alpha'] = st.slider("Mixup Alpha (β dist)", 0.1, 1.0, 0.4, 0.1, key="mixup_alpha")
        apply = st.button("Apply Mixup")
    elif method == 'Spectral Shift':
        st.markdown("**🧪 Parameters for Spectral Shift**")
        params['num_copies'] = st.number_input("Number of shifted copies", 1, 100, 2, 1, key="shift_copies")
        params['max_shift'] = st.slider("Max Shift (±)", 1, 10, 3, key="max_shift")
        apply = st.button("Apply Spectral Shift")
    elif method == 'Gaussian Noise':
        st.markdown("**🧪 Parameters for Gaussian Noise**")
        params['num_copies'] = st.number_input("Noisy copies", 1, 100, 2, 1, key="noise_copies")
        params['std'] = st.number_input("Noise std deviation", 0.001, 1.0, 0.01, 0.001, format="%.3f", key="noise_std")
        apply = st.button("Apply Gaussian Noise")

    # Step 4: Apply augmentation
    if apply:
        augmentor = DataAugmentor(X_current, y_current)

        if method == 'Add Spectra':
            X_new, y_new = augmentor.add_spectra()
        elif method == 'Mixup':
            X_new, y_new = augmentor.mixup(num_copies=params['num_copies'], alpha=params['alpha'])
        elif method == 'Spectral Shift':
            X_new, y_new = augmentor.spectral_shift(num_copies=params['num_copies'], max_shift=params['max_shift'])
        elif method == 'Gaussian Noise':
            X_new, y_new = augmentor.gaussian_noise(num_copies=params['num_copies'], std=params['std'])
        else:
            X_new, y_new = X_current, y_current

        # Save new state
        st.session_state['augmented_X'] = X_new
        st.session_state['augmented_y'] = y_new
        st.session_state['augmentation_history'].append(method)

        st.success(f"✅ {method} applied. New size: {X_new.shape[0]}")

    # Step 5: Visualize spectra
    if 'augmented_X' in st.session_state:
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(min(200, st.session_state['augmented_X'].shape[0])):
            ax.plot(st.session_state['augmented_X'].columns.astype(float),
                    st.session_state['augmented_X'].iloc[i], alpha=0.4)
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Intensity")
        ax.set_title("Current Augmented Spectra")
        st.pyplot(fig)

    # Step 6: Show history and control
    if st.session_state['augmentation_history']:
        st.markdown("### 🧾 Applied Augmentation Steps:")
        for i, step in enumerate(st.session_state['augmentation_history'], 1):
            st.write(f"{i}. {step}")

    if st.button("🔄 Reset Augmentation"):
            st.session_state['augmented_X'] = st.session_state['preprocessed_X'].copy()
            st.session_state['augmented_y'] = st.session_state['preprocessed_y'].copy()
            st.session_state['augmentation_history'] = []
            st.success("Augmentation reset.")
    if st.button("✅ Finalize Augmentation"):
            st.session_state['final_augmented_X'] = st.session_state['augmented_X']
            st.session_state['final_augmented_y'] = st.session_state['augmented_y']
            st.success("Final augmented dataset stored. Ready for modeling or export.")






######################################################################################  Model Trainer ##################################################################################



    st.markdown("##  Train-Test Split")

    # Select test size
    test_size = st.slider("Select test size fraction", 0.1, 0.5, 0.3, 0.05)

    # Use augmented data if available
    if 'final_augmented_X' in st.session_state and 'final_augmented_y' in st.session_state:
        st.success("Using final augmented data for splitting.")

        X = st.session_state['final_augmented_X']
        y = st.session_state['final_augmented_y']

        X_train , X_test , y_train , y_test  = train_test_split(X , y , test_size= test_size , random_state= 42)




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
                auto.plot_model_scores()
                st.success("Automated Regression Model Run Complete")

            else:
                auto.run_classification(X_train, X_test, y_train, y_test)
                st.success("Automated Classification Model Run Complete")

if __name__ == "__main__":
    main()




