import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================================
# PAGE SETUP
# ==========================================================

st.set_page_config(
    page_title="Sensor Characterisation Suite",
    layout="wide"
)

st.title("Sensor Characterisation Suite")

# ==========================================================
# FUNCTIONS
# ==========================================================

def fit_calibration(x, y):

    model = LinearRegression()

    model.fit(
        x.reshape(-1, 1),
        y
    )

    predicted = model.predict(
        x.reshape(-1, 1)
    )

    slope = float(model.coef_[0])

    intercept = float(
        model.intercept_
    )

    r_squared = float(
        r2_score(
            y,
            predicted
        )
    )

    return (
        slope,
        intercept,
        r_squared,
        predicted
    )

# ==========================================================
# USER INPUT
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

r2_limit = st.number_input(
    "Minimum R²",
    min_value=0.0,
    max_value=1.0,
    value=0.9950,
    step=0.0001,
    format="%.4f"
)

# ==========================================================
# PROCESS DATA
# ==========================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        required_columns = [
            "Generated_umol_mol",
            "Sensor_Response_umol_mol"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing columns: {missing_columns}"
            )

            st.stop()

        generated = (
            df["Generated_umol_mol"]
            .astype(float)
            .values
        )

        response = (
            df["Sensor_Response_umol_mol"]
            .astype(float)
            .values
        )

        (
            slope,
            intercept,
            r_squared,
            predicted
        ) = fit_calibration(
            generated,
            response
        )

        residuals = (
            response - predicted
        )

        max_residual = np.max(
            np.abs(residuals)
        )

        span = (
            np.max(generated)
            - np.min(generated)
        )

        if span > 0:
            max_residual_percent_fs = (
                max_residual / span
            ) * 100
        else:
            max_residual_percent_fs = np.nan

        status = (
            "PASS"
            if r_squared >= r2_limit
            else "FAIL"
        )

        summary_df = pd.DataFrame({

            "Metric": [
                "Slope",
                "Intercept",
                "R²",
                "Maximum Residual",
                "Maximum Residual (%FS)",
                "Linearity Status"
            ],

            "Value": [
                slope,
                intercept,
                r_squared,
                max_residual,
                max_residual_percent_fs,
                status
            ]

        })

        linearity_df = pd.DataFrame({

            "Generated_umol_mol":
                generated,

            "Sensor_Response_umol_mol":
                response,

            "Predicted_Response":
                predicted,

            "Residual":
                residuals

        })

        tab1, tab2 = st.tabs([
            "Summary",
            "Linearity"
        ])

        with tab1:

            st.dataframe(
                summary_df,
                use_container_width=True
            )

        with tab2:

            st.dataframe(
                linearity_df,
                use_container_width=True
            )

    except Exception as e:

        st.exception(e)
