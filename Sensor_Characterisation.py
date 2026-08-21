import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import BytesIO

st.set_page_config(
    page_title="Sensor Characterisation Suite",
    layout="wide"
)

st.title("Sensor Characterisation Suite")

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def round_df(df):
    return df.round(3)


def calculate_regression(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    y_pred = model.predict(x.reshape(-1, 1))

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(r2_score(y, y_pred))

    return slope, intercept, r2, y_pred


def calculate_resolution(signal, concentration):
    signal = np.asarray(signal)
    concentration = np.asarray(concentration)

    sensitivity = np.polyfit(concentration, signal, 1)[0]

    noise = np.std(signal - np.polyval(
        np.polyfit(concentration, signal, 1),
        concentration
    ))

    if abs(sensitivity) < 1e-12:
        return np.nan

    resolution = noise / abs(sensitivity)

    return resolution


def calculate_accuracy(expected, measured):
    error = measured - expected
    abs_error = np.abs(error)

    return pd.DataFrame({
        "Expected_umol_mol": expected,
        "Measured_umol_mol": measured,
        "Error_umol_mol": error,
        "Absolute_Error_umol_mol": abs_error
    })


def calculate_reversibility(df):

    grouped = df.groupby("Expected_umol_mol")

    rows = []

    for conc, grp in grouped:

        if len(grp) >= 2:

            first = grp["Calculated_umol_mol"].iloc[0]
            last = grp["Calculated_umol_mol"].iloc[-1]

            diff = abs(last - first)

            rows.append([
                conc,
                first,
                last,
                diff
            ])

    result = pd.DataFrame(
        rows,
        columns=[
            "Expected_umol_mol",
            "First_Measurement",
            "Last_Measurement",
            "Difference"
        ]
    )

    return result


def calculate_hysteresis(df):

    grouped = df.groupby("Expected_umol_mol")

    rows = []

    for conc, grp in grouped:

        if len(grp) >= 2:

            hyst = (
                grp["Calculated_umol_mol"].max()
                - grp["Calculated_umol_mol"].min()
            )

            rows.append([
                conc,
                hyst
            ])

    result = pd.DataFrame(
        rows,
        columns=[
            "Expected_umol_mol",
            "Hysteresis"
        ]
    )

    return result


def build_excel_export(sheets):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for name, df in sheets.items():
            df.to_excel(
                writer,
                sheet_name=name[:31],
                index=False
            )

    output.seek(0)
    return output


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

full_scale = st.number_input(
    "Full Scale Concentration (umol/mol)",
    min_value=1.0,
    value=100.0,
    step=1.0
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

if uploaded_file:

    # --------------------------------------------------------------
    # Read file
    # --------------------------------------------------------------

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    required_cols = [
        "Expected_umol_mol",
        "Signal_mA"
    ]

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        st.error(
            f"Missing columns: {missing}"
        )
        st.stop()

    df = df.copy()

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------

    x = df["Expected_umol_mol"].astype(float).values
    y = df["Signal_mA"].astype(float).values

    slope, intercept, r2, y_fit = calculate_regression(x, y)

    calculated_conc = (y - intercept) / slope

    df["Calculated_umol_mol"] = calculated_conc

    # --------------------------------------------------------------
    # Linearity
    # --------------------------------------------------------------

    linearity_df = pd.DataFrame({
        "Expected_umol_mol": x,
        "Signal_mA": y,
        "Predicted_mA": y_fit,
        "Residual_mA": y - y_fit
    })

    max_residual = np.max(
        np.abs(linearity_df["Residual_mA"])
    )

    linearity_percent_fs = (
        max_residual /
        (np.max(y_fit) - np.min(y_fit))
    ) * 100

    # --------------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------------

    accuracy_df = calculate_accuracy(
        df["Expected_umol_mol"],
        df["Calculated_umol_mol"]
    )

    max_accuracy_error = accuracy_df[
        "Absolute_Error_umol_mol"
    ].max()

    accuracy_percent_fs = (
        max_accuracy_error /
        full_scale
    ) * 100

    # --------------------------------------------------------------
    # Resolution
    # --------------------------------------------------------------

    resolution = calculate_resolution(
        df["Signal_mA"],
        df["Expected_umol_mol"]
    )

    resolution_df = pd.DataFrame({
        "Resolution_umol_mol": [resolution]
    })

    # --------------------------------------------------------------
    # Reversibility
    # --------------------------------------------------------------

    reversibility_df = calculate_reversibility(df)

    # --------------------------------------------------------------
    # Hysteresis
    # --------------------------------------------------------------

    hysteresis_df = calculate_hysteresis(df)

    if len(hysteresis_df):
        max_hysteresis = hysteresis_df[
            "Hysteresis"
        ].max()
    else:
        max_hysteresis = np.nan

    # --------------------------------------------------------------
    # Pass Fail
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary_df = pd.DataFrame({
        "Metric": [
            "Slope",
            "Intercept",
            "R²",
            "Max Accuracy Error",
            "Accuracy %FS",
            "Linearity %FS",
            "Resolution",
            "Max Hysteresis"
        ],
        "Value": [
            slope,
            intercept,
            r2,
            max_accuracy_error,
            accuracy_percent_fs,
            linearity_percent_fs,
            resolution,
            max_hysteresis
        ]
    })

    summary_df = round_df(summary_df)

    # --------------------------------------------------------------
    # Tabs
    # --------------------------------------------------------------

    tabs = st.tabs([
        "Summary",
        "Linearity",
        "Accuracy",
        "Resolution",
        "Reversibility",
        "Hysteresis",
        "Pass / Fail",
        "Methods & Formulae"
    ])

    with tabsst.dataframe(summary_df)

    with tabsst.dataframe(round_df(linearity_df))

    with tabsst.dataframe(round_df(accuracy_df))

    with tabsst.dataframe(round_df(resolution_df))

    with tabsif len(reversibility_df):
            st.dataframe(round_df(reversibility_df))
        else:
            st.info(
                "Reversibility requires repeated concentrations."
            )

    with tabsif len(hysteresis_df):
            st.dataframe(round_df(hysteresis_df))
        else:
            st.info(
                "Hysteresis requires repeated concentrations."
            )

    with tabsst.dataframe(round_df(pass_fail_df))

    with tabsst.markdown("""
### Linearity

Linear regression:

Signal = m × Concentration + c

### Accuracy

Accuracy Error:

Measured − Expected

%FS:

(Error / Full Scale) × 100

### Resolution

Resolution:

Noise / Sensitivity

### Reversibility

Difference between repeated measurements at the same concentration.

### Hysteresis

Maximum spread of repeated measurements at the same concentration.

### Pass / Fail

Compared against user-entered acceptance limits.
""")

    # --------------------------------------------------------------
    # Excel Export
    # --------------------------------------------------------------

    excel_file = build_excel_export({
        "Summary": round_df(summary_df),
        "Linearity": round_df(linearity_df),
        "Accuracy": round_df(accuracy_df),
        "Resolution": round_df(resolution_df),
        "Reversibility": round_df(reversibility_df),
        "Hysteresis": round_df(hysteresis_df),
        "Pass_Fail": round_df(pass_fail_df)
    })

    st.download_button(
        "Download Excel Report",
        excel_file,
        file_name="Sensor_Characterisation_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
