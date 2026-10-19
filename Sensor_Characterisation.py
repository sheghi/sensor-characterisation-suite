import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

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

    y_pred = model.predict(
        x.reshape(-1, 1)
    )

    slope = float(model.coef_[0])

    intercept = float(model.intercept_)

    r2 = float(
        r2_score(
            y,
            y_pred
        )
    )

    return slope, intercept, r2, y_pred


def create_excel_report(sheet_dict):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for name, sheet in sheet_dict.items():

            sheet.to_excel(
                writer,
                sheet_name=name[:31],
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

r2_limit = st.number_input(
    "Minimum R²",
    value=0.9950,
    step=0.0001,
    format="%.4f"
)

accuracy_limit = st.number_input(
    "Maximum Accuracy (%)",
    value=2.0
)

precision_limit = st.number_input(
    "Maximum Precision (%RSD)",
    value=2.0
)

hysteresis_limit = st.number_input(
    "Maximum Hysteresis (µmol/mol)",
    value=1.0
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

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            st.error(
                f"Missing columns: {missing}"
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

        # ==================================================
        # LINEARITY
        # ==================================================

        slope, intercept, r_squared, predicted = (
            fit_calibration(
                generated,
                response
            )
        )

        residuals = response - predicted

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

        # ==================================================
        # ACCURACY
        # ==================================================

        accuracy_df = pd.DataFrame({

            "Generated_umol_mol":
                generated,

            "Sensor_Response_umol_mol":
                response

        })

        accuracy_df["Error_umol_mol"] = (
            accuracy_df[
                "Sensor_Response_umol_mol"
            ]
            -
            accuracy_df[
                "Generated_umol_mol"
            ]
        )

        accuracy_df["Accuracy_%"] = np.where(

            accuracy_df[
                "Generated_umol_mol"
            ] > 0,

            np.abs(
                accuracy_df[
                    "Error_umol_mol"
                ]
            )
            /
            accuracy_df[
                "Generated_umol_mol"
            ]
            * 100,

            np.nan

        )

        max_accuracy = (
            accuracy_df[
                "Accuracy_%"
            ].max()
        )

        bias = (
            accuracy_df[
                "Error_umol_mol"
            ].mean()
        )

        # ==================================================
        # PRECISION
        # ==================================================

        precision_rows = []

        for conc, group in df.groupby(
            "Generated_umol_mol"
        ):

            if len(group) > 1:

                mean_val = group[
                    "Sensor_Response_umol_mol"
                ].mean()

                sd = group[
                    "Sensor_Response_umol_mol"
                ].std(ddof=1)

                rsd = (
                    sd / mean_val * 100
                    if mean_val != 0
                    else np.nan
                )

                precision_rows.append({

                    "Generated_umol_mol":
                        conc,

                    "Mean":
                        mean_val,

                    "SD":
                        sd,

                    "RSD_%":
                        rsd

                })

        precision_df = pd.DataFrame(
            precision_rows
        )

        max_rsd = (
            precision_df["RSD_%"].max()
            if len(precision_df)
            else np.nan
        )

        # ==================================================
        # HYSTERESIS
        # ==================================================

        hysteresis_rows = []

        for conc, group in df.groupby(
            "Generated_umol_mol"
        ):

            if len(group) > 1:

                hyst = (
                    group[
                        "Sensor_Response_umol_mol"
                    ].max()
                    -
                    group[
                        "Sensor_Response_umol_mol"
                    ].min()
                )

                hysteresis_rows.append({

                    "Generated_umol_mol":
                        conc,

                    "Hysteresis_umol_mol":
                        hyst

                })

        hysteresis_df = pd.DataFrame(
            hysteresis_rows
        )

        max_hysteresis = (
            hysteresis_df[
                "Hysteresis_umol_mol"
            ].max()
            if len(hysteresis_df)
            else np.nan
        )

        # ==================================================
        # PASS / FAIL
        # ==================================================

        pass_fail_df = pd.DataFrame({

            "Parameter": [
                "Linearity",
                "Accuracy",
                "Precision",
                "Hysteresis"
            ],

            "Result": [
                r_squared,
                max_accuracy,
                max_rsd,
                max_hysteresis
            ],

            "Limit": [
                r2_limit,
                accuracy_limit,
                precision_limit,
                hysteresis_limit
            ]

        })

        pass_fail_df["Status"] = [

            "PASS" if r_squared >= r2_limit else "FAIL",

            "PASS" if max_accuracy <= accuracy_limit else "FAIL",

            "PASS" if (
                pd.notna(max_rsd)
                and max_rsd <= precision_limit
            ) else "FAIL",

            "PASS" if (
                pd.notna(max_hysteresis)
                and max_hysteresis <= hysteresis_limit
            ) else "FAIL"
        ]

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
                "Maximum Accuracy (%)",
                "Bias (µmol/mol)",
                "Maximum RSD (%)",
                "Maximum Hysteresis",
                "Overall Status"

            ],

            "Value": [

                slope,
                intercept,
                r_squared,
