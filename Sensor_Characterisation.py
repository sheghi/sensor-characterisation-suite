import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ============================================================
# Page Setup
# ============================================================

st.set_page_config(
    page_title="Sensor Characterisation Suite",
    layout="wide"
)

st.title("Sensor Characterisation Suite")

# ============================================================
# Helper Functions
# ============================================================

def round_df(df):
    return df.round(3)


def perform_regression(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    y_pred = model.predict(x.reshape(-1, 1))

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(r2_score(y, y_pred))

    return slope, intercept, r2, y_pred


def calculate_resolution(signal, concentration):

    coeffs = np.polyfit(concentration, signal, 1)

    sensitivity = coeffs[0]

    fitted = np.polyval(coeffs, concentration)

    noise = np.std(signal - fitted)

    if abs(sensitivity) < 1e-12:
        return np.nan

    return noise / abs(sensitivity)


def create_excel_report(sheet_dict):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for sheet_name, df in sheet_dict.items():
            df.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False
            )

    output.seek(0)

    return output


# ============================================================
# User Inputs
# ============================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

full_scale = st.number_input(
    "Full Scale Concentration (µmol/mol)",
    min_value=1.0,
    value=100.0
)

accuracy_limit = st.number_input(
    "Accuracy Limit (%FS)",
    min_value=0.0,
    value=2.0
)

linearity_limit = st.number_input(
    "Linearity Limit (%FS)",
    min_value=0.0,
    value=2.0
)

# ============================================================
# Data Processing
# ============================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        required_columns = [
            "Expected_umol_mol",
            "Signal_mA"
        ]

        missing = [
            c for c in required_columns
            if c not in df.columns
        ]

        if missing:
            st.error(
                f"Missing required columns: {missing}"
            )
            st.stop()

        # ----------------------------------------------------

        x = df["Expected_umol_mol"].astype(float).values
        y = df["Signal_mA"].astype(float).values

        slope, intercept, r2, predicted_signal = (
            perform_regression(x, y)
        )

        calculated_concentration = (
            y - intercept
        ) / slope

        df["Calculated_umol_mol"] = (
            calculated_concentration
        )

        # ====================================================
        # Linearity
        # ====================================================

        residuals = y - predicted_signal

        linearity_df = pd.DataFrame({
            "Expected_umol_mol": x,
            "Signal_mA": y,
            "Predicted_mA": predicted_signal,
            "Residual_mA": residuals
        })

        signal_span = (
            predicted_signal.max()
            - predicted_signal.min()
        )

        max_residual = np.abs(
            residuals
        ).max()

        if signal_span > 0:
            linearity_percent_fs = (
                max_residual / signal_span
            ) * 100
        else:
            linearity_percent_fs = np.nan

        # ====================================================
        # Accuracy
        # ====================================================

        accuracy_df = pd.DataFrame({
            "Expected_umol_mol": x,
            "Measured_umol_mol":
                calculated_concentration
        })

        accuracy_df["Error_umol_mol"] = (
            accuracy_df["Measured_umol_mol"]
            - accuracy_df["Expected_umol_mol"]
        )

        accuracy_df["Absolute_Error_umol_mol"] = (
            accuracy_df["Error_umol_mol"].abs()
        )

        max_accuracy_error = (
            accuracy_df[
                "Absolute_Error_umol_mol"
            ].max()
        )

        accuracy_percent_fs = (
            max_accuracy_error / full_scale
        ) * 100

        # ====================================================
        # Resolution
        # ====================================================

        resolution_value = calculate_resolution(
            y,
            x
        )

        resolution_df = pd.DataFrame({
            "Resolution_umol_mol":
                [resolution_value]
        })

        # ====================================================
        # Reversibility
        # ====================================================

        reversibility_rows = []

        for conc, grp in df.groupby(
            "Expected_umol_mol"
        ):

            if len(grp) >= 2:

                first = grp[
                    "Calculated_umol_mol"
                ].iloc[0]

                last = grp[
                    "Calculated_umol_mol"
                ].iloc[-1]

                reversibility_rows.append([
                    conc,
                    first,
                    last,
                    abs(last - first)
                ])

        reversibility_df = pd.DataFrame(
            reversibility_rows,
            columns=[
                "Expected_umol_mol",
                "First",
                "Last",
                "Difference"
            ]
        )

        # ====================================================
        # Hysteresis
        # ====================================================

        hysteresis_rows = []

        for conc, grp in df.groupby(
            "Expected_umol_mol"
        ):

            if len(grp) >= 2:

                spread = (
                    grp[
                        "Calculated_umol_mol"
                    ].max()
                    -
                    grp[
                        "Calculated_umol_mol"
                    ].min()
                )

                hysteresis_rows.append([
                    conc,
                    spread
                ])

        hysteresis_df = pd.DataFrame(
            hysteresis_rows,
            columns=[
                "Expected_umol_mol",
                "Hysteresis"
            ]
        )

        # ====================================================
        # Pass / Fail
        # ====================================================

        pass_fail_df = pd.DataFrame({
            "Parameter": [
                "Linearity",
                "Accuracy"
            ],
            "Result_%FS": [
                linearity_percent_fs,
                accuracy_percent_fs
            ],
            "Limit_%FS": [
                linearity_limit,
                accuracy_limit
            ]
        })

        pass_fail_df["Status"] = np.where(
            pass_fail_df["Result_%FS"]
            <= pass_fail_df["Limit_%FS"],
            "PASS",
            "FAIL"
        )

        # ====================================================
        # Summary
        # ====================================================

        summary_df = pd.DataFrame({
            "Metric": [
                "Slope",
                "Intercept",
                "R²",
                "Linearity %FS",
                "Accuracy %FS",
                "Resolution"
            ],
            "Value": [
                slope,
                intercept,
                r2,
                linearity_percent_fs,
                accuracy_percent_fs,
                resolution_value
            ]
        })

        summary_df = round_df(summary_df)

        # ====================================================
        # Tabs
        # ====================================================

        (
            tab1,
            tab2,
            tab3,
            tab4,
            tab5,
            tab6,
            tab7,
            tab8
        ) = st.tabs([
            "Summary",
            "Linearity",
            "Accuracy",
            "Resolution",
            "Reversibility",
            "Hysteresis",
            "Pass / Fail",
            "Methods & Formulae"
        ])

        with tab1:
            st.dataframe(summary_df)

        with tab2:
            st.dataframe(
                round_df(linearity_df)
            )

        with tab3:
            st.dataframe(
                round_df(accuracy_df)
            )

        with tab4:
            st.dataframe(
                round_df(resolution_df)
            )

        with tab5:

            if len(reversibility_df):
                st.dataframe(
                    round_df(reversibility_df)
                )
            else:
                st.info(
                    "No repeated points available."
                )

        with tab6:

            if len(hysteresis_df):
                st.dataframe(
                    round_df(hysteresis_df)
                )
            else:
                st.info(
                    "No repeated points available."
                )

        with tab7:
            st.dataframe(
                round_df(pass_fail_df)
            )

        with tab8:

            st.markdown("""
### Linearity

Signal = m × Concentration + c

---

### Accuracy

Error = Measured − Expected

Accuracy (%FS)

(Error / Full Scale) × 100

---

### Resolution

Resolution = Noise / Sensitivity

---

### Reversibility

Difference between first and last measurement at the same concentration.

---

### Hysteresis

Maximum spread between repeated measurements at the same concentration.

---

### Pass / Fail

Results are compared against user-defined limits.
""")

        # ====================================================
        # Excel Export
        # ====================================================

        excel_file = create_excel_report({

            "Summary":
                round_df(summary_df),

            "Linearity":
                round_df(linearity_df),

            "Accuracy":
                round_df(accuracy_df),

            "Resolution":
                round_df(resolution_df),

            "Reversibility":
                round_df(reversibility_df),

            "Hysteresis":
                round_df(hysteresis_df),

            "PassFail":
                round_df(pass_fail_df)

        })

        st.download_button(
            "Download Excel Report",
            data=excel_file,
            file_name="Sensor_Characterisation_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.exception(e)
