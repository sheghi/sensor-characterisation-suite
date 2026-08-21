import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Sensor Characterisation Suite V2",
    layout="wide"
)

st.title("Sensor Characterisation Suite V2")

# ==========================================================
# FUNCTIONS
# ==========================================================

def round_df(df):
    return df.round(3)


def fit_calibration(expected, signal):

    model = LinearRegression()
    model.fit(expected.reshape(-1, 1), signal)

    predicted_signal = model.predict(
        expected.reshape(-1, 1)
    )

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r_squared = float(
        r2_score(signal, predicted_signal)
    )

    return (
        slope,
        intercept,
        r_squared,
        predicted_signal
    )


def calculate_resolution(signal, expected):

    coeffs = np.polyfit(
        expected,
        signal,
        1
    )

    sensitivity = coeffs[0]

    fitted_signal = np.polyval(
        coeffs,
        expected
    )

    noise = np.std(
        signal - fitted_signal
    )

    if abs(sensitivity) < 1e-12:
        return np.nan

    return noise / abs(sensitivity)


def create_excel_report(sheet_dict):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for sheet_name, sheet_df in sheet_dict.items():

            sheet_df.to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False
            )

    output.seek(0)

    return output


# ==========================================================
# USER SETTINGS
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

full_scale = st.number_input(
    "Full Scale Concentration (umol/mol)",
    min_value=1.0,
    value=100.0,
    step=1.0
)

accuracy_limit = st.number_input(
    "Accuracy Limit (%)",
    min_value=0.0,
    value=2.0,
    step=0.1
)

linearity_limit = st.number_input(
    "Linearity Limit (%FS)",
    min_value=0.0,
    value=2.0,
    step=0.1
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

        required_columns = [
            "Expected_umol_mol",
            "Signal_mA"
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
        # INPUT DATA
        # --------------------------------------------------

        expected = (
            df["Expected_umol_mol"]
            .astype(float)
            .values
        )

        signal = (
            df["Signal_mA"]
            .astype(float)
            .values
        )

        # --------------------------------------------------
        # CALIBRATION
        # --------------------------------------------------

        (
            slope,
            intercept,
            r_squared,
            predicted_signal
        ) = fit_calibration(
            expected,
            signal
        )

        calculated_concentration = (
            signal - intercept
        ) / slope

        df["Calculated_umol_mol"] = (
            calculated_concentration
        )

        # ==================================================
        # LINEARITY
        # ==================================================

        residuals = (
            signal - predicted_signal
        )

        max_residual = np.max(
            np.abs(residuals)
        )

        signal_span = (
            np.max(predicted_signal)
            -
            np.min(predicted_signal)
        )

        if signal_span == 0:
            linearity_percent_fs = np.nan
        else:
            linearity_percent_fs = (
                max_residual
                / signal_span
            ) * 100

        linearity_df = pd.DataFrame({
            "Expected_umol_mol":
                expected,
            "Signal_mA":
                signal,
            "Predicted_mA":
                predicted_signal,
            "Residual_mA":
                residuals
        })

        # ==================================================
        # ACCURACY
        # ==================================================

        accuracy_df = pd.DataFrame({
            "Expected_umol_mol":
                expected,
            "Calculated_umol_mol":
                calculated_concentration
        })

        accuracy_df["Error_umol_mol"] = (
            accuracy_df[
                "Calculated_umol_mol"
            ]
            -
            accuracy_df[
                "Expected_umol_mol"
            ]
        )

        accuracy_df["Accuracy_%"] = np.where(
            accuracy_df[
                "Expected_umol_mol"
            ] != 0,
            (
                np.abs(
                    accuracy_df[
                        "Error_umol_mol"
                    ]
                )
                /
                accuracy_df[
                    "Expected_umol_mol"
                ]
            ) * 100,
            np.nan
        )

        max_accuracy_percent = (
            accuracy_df["Accuracy_%"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .max()
        )

        # ==================================================
        # RESOLUTION
        # ==================================================

        resolution_value = (
            calculate_resolution(
                signal,
                expected
            )
        )

        resolution_df = pd.DataFrame({
            "Resolution_umol_mol":
                [resolution_value]
        })

        # ==================================================
        # REVERSIBILITY
        # ==================================================

        reversibility_rows = []

        for conc, group in df.groupby(
            "Expected_umol_mol"
        ):

            if len(group) >= 2:

                first_value = (
                    group[
                        "Calculated_umol_mol"
                    ]
                    .iloc[0]
                )

                last_value = (
                    group[
                        "Calculated_umol_mol"
                    ]
                    .iloc[-1]
                )

                difference = abs(
                    last_value - first_value
                )

                reversibility_rows.append([
                    conc,
                    first_value,
                    last_value,
                    difference
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

        # ==================================================
        # HYSTERESIS
        # ==================================================

        hysteresis_rows = []

        for conc, group in df.groupby(
            "Expected_umol_mol"
        ):

            if len(group) >= 2:

                hysteresis = (
                    group[
                        "Calculated_umol_mol"
                    ].max()
                    -
                    group[
                        "Calculated_umol_mol"
                    ].min()
                )

                hysteresis_rows.append([
                    conc,
                    hysteresis
                ])

        hysteresis_df = pd.DataFrame(
            hysteresis_rows,
            columns=[
                "Expected_umol_mol",
                "Hysteresis"
            ]
        )

        # ==================================================
        # PASS / FAIL
        # ==================================================

        pass_fail_df = pd.DataFrame({

            "Parameter": [
                "Linearity",
                "Accuracy"
            ],

            "Result": [
                linearity_percent_fs,
                max_accuracy_percent
            ],

            "Limit": [
                linearity_limit,
                accuracy_limit
            ]

        })

        pass_fail_df["Status"] = np.where(
            pass_fail_df["Result"]
            <=
            pass_fail_df["Limit"],
            "PASS",
            "FAIL"
        )

        overall_status = (
            "PASS"
            if (
                pass_fail_df["Status"]
                == "PASS"
            ).all()
            else "FAIL"
        )

        # ==================================================
        # SUMMARY
        # ==================================================

        summary_df = pd.DataFrame({

            "Metric": [
                "Slope",
                "Intercept",
                "R²",
                "Maximum Accuracy %",
                "Linearity %FS",
                "Resolution (umol/mol)",
                "Overall Status"
            ],

            "Value": [
                slope,
                intercept,
                r_squared,
                max_accuracy_percent,
                linearity_percent_fs,
                resolution_value,
                overall_status
            ]

        })

        # ==================================================
        # TABS
        # ==================================================

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
            st.dataframe(
                round_df(summary_df)
            )

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
                    round_df(
                        reversibility_df
                    )
                )
            else:
                st.info(
                    "No repeated concentrations available."
                )

        with tab6:

            if len(hysteresis_df):
                st.dataframe(
                    round_df(
                        hysteresis_df
                    )
                )
            else:
                st.info(
                    "No repeated concentrations available."
                )

        with tab7:
            st.dataframe(
                round_df(pass_fail_df)
            )

        with tab8:

            st.markdown(
                """
### Linearity

Linear Regression:

**Signal = m × Concentration + c**

Linearity (%FS)

**(Maximum Residual / Signal Span) × 100**

---

### Accuracy

Calculated Concentration:

**(Signal − Intercept) / Slope**

Error:

**Calculated − Expected**

Accuracy (%):

**|Error| / Expected × 100**

---

### Resolution

Resolution:

**Noise / Sensitivity**

Where:

- Noise = Standard Deviation of Residuals
- Sensitivity = Regression Slope

---

### Reversibility

**|Last Measurement − First Measurement|**

at identical concentration.

---

### Hysteresis

**Maximum Reading − Minimum Reading**

at identical concentration.

---

### Pass / Fail

Based on:

- Maximum Accuracy %
- Linearity %FS

against user-defined limits.
"""
            )

        # ==================================================
        # EXCEL EXPORT
        # ==================================================

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
            label="Download Excel Report",
            data=excel_file,
            file_name="Sensor_Characterisation_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.exception(e)
