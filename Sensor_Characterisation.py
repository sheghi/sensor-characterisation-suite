import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Sensor Characterisation Suite V3",
    layout="wide"
)

st.title("Sensor Characterisation Suite V3")

# ==========================================================
# FUNCTIONS
# ==========================================================

def round_sig(x, sig=4):
    """
    Round values to specified significant figures.
    """

    if pd.isna(x):
        return x

    if isinstance(x, str):
        return x

    if x == 0:
        return 0

    return float(f"{x:.{sig}g}")


def round_df(df):
    """
    Apply significant figure rounding to all numeric
    values in a dataframe.
    """

    return df.applymap(
        lambda x:
        round_sig(x, 4)
        if isinstance(
            x,
            (int, float, np.number)
        )
        else x
    )


def fit_calibration(
    generated_amount_fraction,
    sensor_response
):
    """
    Perform linear regression and return:
    slope, intercept, R² and predicted response.
    """

    model = LinearRegression()

    model.fit(
        generated_amount_fraction.reshape(-1, 1),
        sensor_response
    )

    predicted_response = model.predict(
        generated_amount_fraction.reshape(-1, 1)
    )

    slope = float(model.coef_[0])

    intercept = float(model.intercept_)

    r_squared = float(
        r2_score(
            sensor_response,
            predicted_response
        )
    )

    return (
        slope,
        intercept,
        r_squared,
        predicted_response
    )

# ==========================================================
# USER SETTINGS
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

r2_limit = st.number_input(
    "Minimum R² Acceptance Criterion",
    min_value=0.0000,
    max_value=1.0000,
    value=0.9950,
    step=0.0001,
    format="%.4f"
)

# ==========================================================
# PROCESS DATA
# ==========================================================

if uploaded_file:

    try:

        # --------------------------------------------------
        # READ FILE
        # --------------------------------------------------

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # --------------------------------------------------
        # REQUIRED COLUMNS
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
            predicted_response

        ) = fit_calibration(
            generated,
            response
        )

        residuals = (
            response
            -
            predicted_response
        )

        max_residual = np.max(
            np.abs(residuals)
        )

        span = (
            np.max(generated)
            -
            np.min(generated)
        )

        if span == 0:

            max_residual_percent_fs = np.nan

        else:

            max_residual_percent_fs = (
                max_residual
                /
                span
            ) * 100

        # --------------------------------------------------
        # LINEARITY TABLE
        # --------------------------------------------------

        linearity_df = pd.DataFrame({

            "Generated_umol_mol":
                generated,

            "Sensor_Response_umol_mol":
                response,

            "Predicted_Response_umol_mol":
                predicted_response,

            "Residual_umol_mol":
                residuals,

            "Absolute_Residual_umol_mol":
                np.abs(residuals)

        })

        # --------------------------------------------------
        # PASS / FAIL
        # --------------------------------------------------

        linearity_status = (
            "PASS"
            if r_squared >= r2_limit
            else "FAIL"
        )

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
                linearity_status
            ]

        })

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        summary_df = pd.DataFrame({

            "Metric": [

                "Slope",
                "Intercept",
                "R²",
                "Maximum Residual (umol/mol)",
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
        # DISPLAY
        # --------------------------------------------------

        tab1, tab2, tab3 = st.tabs([
            "Summary",
            "Linearity Data",
            "Methods"
        ])

        with tab1:

            st.subheader(
                "Calibration Summary"
            )

            st.dataframe(
                round_df(summary_df),
                use_container_width=True
            )

            st.subheader(
                "Linearity Assessment"
            )

            st.dataframe(
                round_df(pass_fail_df),
                use_container_width=True
            )

        with tab2:

            st.dataframe(
                round_df(linearity_df),
                use_container_width=True
            )

        with tab3:

            st.markdown(
                """
### Calibration Model

Linear regression is performed using:

**Sensor Response = m × Generated Amount Fraction + c**

Where:

- m = slope
- c = intercept

---

### Linearity Assessment

Linearity is assessed using the coefficient of determination:

**R²**

Acceptance criterion:

**R² ≥ User Defined Limit**

---

### Residual

Residual =

**Sensor Response − Predicted Response**

---

### Maximum Residual

Maximum Residual =

**max(|Residual|)**

---

### Maximum Residual (%FS)

**(Maximum Absolute Residual / Concentration Span) × 100**

This provides an additional indication of non-linearity across the calibration range.
                """
            )

    except Exception as e:

        st.exception(e)
