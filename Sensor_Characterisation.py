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

def round_sig(x, sig=4):

    if pd.isna(x):
        return x

    if isinstance(x, str):
        return x

    if x == 0:
        return 0

    return float(f"{x:.{sig}g}")


def round_df(df):

    df_copy = df.copy()

    for col in df_copy.columns:

        if pd.api.types.is_numeric_dtype(df_copy[col]):

            df_copy[col] = df_copy[col].apply(
                lambda x: round_sig(x, 4)
            )

    return df_copy


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
    "Minimum Acceptable R²",
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

        # --------------------------------------------------
        # READ FILE
        # --------------------------------------------------

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        # --------------------------------------------------
        # CHECK COLUMNS
        # --------------------------------------------------

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

        # --------------------------------------------------
        # EXTRACT DATA
        # --------------------------------------------------

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

        # --------------------------------------------------
        # LINEAR REGRESSION
        # --------------------------------------------------

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
            -
            np.min(generated)
        )

        if span > 0:

            max_residual_percent_fs = (
                max_residual / span
            ) * 100

        else:

            max_residual_percent_fs = np.nan

        # --------------------------------------------------
        # PASS / FAIL
        # --------------------------------------------------

        linearity_status = (
            "PASS"
            if r_squared >= r2_limit
            else "FAIL"
        )

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        summary_df = pd.DataFrame({

            "Metric": [

                "Slope",
                "Intercept",
                "R²",
                "Maximum Residual (µmol/mol)",
                "Maximum Residual (%FS)",
                "Linearity Status"

            ],

            "Value": [

                slope,
                intercept,
                r_squared,
                max_residual,
                max_residual_percent_fs,
                linearity_status

            ]

        })

        # --------------------------------------------------
        # LINEARITY TABLE
        # --------------------------------------------------

        linearity_df = pd.DataFrame({

            "Generated_umol_mol":
                generated,

            "Sensor_Response_umol_mol":
                response,

            "Predicted_Response_umol_mol":
                predicted,

            "Residual_umol_mol":
                residuals

        })

        # --------------------------------------------------
        # CALIBRATION PLOT DATA
        # --------------------------------------------------

        plot_points = pd.DataFrame({

            "Generated_umol_mol":
                generated,

            "Sensor_Response_umol_mol":
                response

        })

        plot_fit = pd.DataFrame({

            "Generated_umol_mol":
                generated,

            "Regression_Fit":
                predicted

        })

        plot_fit = plot_fit.sort_values(
            "Generated_umol_mol"
        )

        # --------------------------------------------------
        # TABS
        # --------------------------------------------
