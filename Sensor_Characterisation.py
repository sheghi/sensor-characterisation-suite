import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import plotly.graph_objects as go


# ==========================================================
# PAGE CONFIGURATION
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

    if x == 0:
        return 0

    return float(f"{x:.{sig}g}")


def round_df(df):

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].apply(
                lambda x: round_sig(x, 4)
            )

    return df


def fit_calibration(x, y):

    model = LinearRegression()

    model.fit(
        x.reshape(-1, 1),
        y
    )

    y_pred = model.predict(
        x.reshape(-1, 1)
    )

    slope = float(
        model.coef_[0]
    )

    intercept = float(
        model.intercept_
    )

    r2 = float(
        r2_score(
            y,
            y_pred
        )
    )

    return (
        slope,
        intercept,
        r2,
        y_pred
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
# MAIN
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

        if len(missing_columns) > 0:

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
            .to_numpy()
        )

        response = (
            df["Sensor_Response_umol_mol"]
            .astype(float)
            .to_numpy()
        )

        # --------------------------------------------------
        # REGRESSION
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

        abs_residuals = np.abs(
            residuals
        )

        max_residual = np.max(
            abs_residuals
        )

        calibration_span = (
            np.max(generated)
            - np.min(generated)
        )

        if calibration_span > 0:

            max_residual_percent_fs = (
                max_residual
                / calibration_span
            ) * 100

        else:

            max_residual_percent_fs = np.nan

        # --------------------------------------------------
        # PASS FAIL
        # --------------------------------------------------

        status = (
            "PASS"
            if r_squared >= r2_limit
            else "FAIL"
        )

        # --------------------------------------------------
        # TABLES
        # --------------------------------------------------

        summary_df = pd.DataFrame({

            "Metric": [
                "Slope",
                "Intercept",
                "R²",
                "Maximum Residual (µmol/mol)",
                "Maximum Residual (%FS)",
                "Status"
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

        pass_fail_df = pd.DataFrame({

            "Parameter": [
                "Linearity (R²)"
            ],

            "Result": [
                r_squared
            ],

            "Acceptance Limit": [
                r2_limit
            ],

            "Status": [
                status
            ]

        })

        linearity_df = pd.DataFrame({

            "Generated_umol_mol":
                generated,

            "Sensor_Response_umol_mol":
                response,

            "Predicted_Response_umol_mol":
                predicted,

            "Residual_umol_mol":
                residuals,

            "Absolute_Residual_umol_mol":
                abs_residuals

        })

        # --------------------------------------------------
        # TABS
        # --------------------------------------------------

        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary",
            "Calibration Plot",
            "Linearity Data",
            "Methods"
        ])

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        with tab1:

            st.subheader(
                "Calibration Summary"
            )

            st.dataframe(
                round_df(summary_df),
                use_container_width=True
            )

            st.subheader(
                "Pass / Fail"
            )

            st.dataframe(
                round_df(pass_fail_df),
                use_container_width=True
            )

        # --------------------------------------------------
        # CALIBRATION PLOT
        # --------------------------------------------------

        with tab2:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.scatter(
                generated,
                response,
                color="blue",
                label="Measured Data"
            )

            order = np.argsort(
                generated
            )

            ax.plot(
                generated[order],
                predicted[order],
                color="red",
                linewidth=2,
                label="Linear Fit"
            )

            ax.set_xlabel(
                "Generated Amount Fraction (µmol/mol)"
            )

            ax.set_ylabel(
                "Sensor Response (µmol/mol)"
            )

            ax.set_title(
                "Calibration Curve"
            )

            text = (
                f"y = {slope:.4g}x + {intercept:.4g}\n"
                f"R² = {r_squared:.4g}"
            )

            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(
                    facecolor="white",
                    alpha=0.8
                )
            )

            ax.grid(True)

            ax.legend()

            st.pyplot(fig)

        # --------------------------------------------------
        # DATA
        # --------------------------------------------------

        with tab3:

            st.dataframe(
                round_df(linearity_df),
                use_container_width=True
            )

        # --------------------------------------------------
        # METHODS
        # --------------------------------------------------

        with tab4:

            st.markdown("""
### Calibration Model

Sensor Response = m × Generated Amount Fraction + c

where:

- m = slope
- c = intercept

### Linearity Criterion

R² is used to assess linearity.

Pass criterion:

R² ≥ User-defined acceptance limit

### Residual

Residual = Sensor Response − Predicted Response

### Maximum Residual (%FS)

(Maximum Absolute Residual / Calibration Span) × 100
""")

    except Exception as e:

        st.exception(e)
