import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

    return round(
        x,
        sig - int(np.floor(np.log10(abs(x)))) - 1
    )


def round_df(df):

    df_copy = df.copy()

    for col in df_copy.columns:

        if pd.api.types.is_numeric_dtype(
            df_copy[col]
        ):

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
# USER INPUTS
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

r2_limit = st.number_input(
    "Minimum Acceptable R²",
    value=0.9950,
    step=0.0001,
    format="%.4f"
)

precision_limit = st.number_input(
    "Maximum Precision (%RSD)",
    value=2.0
)

bias_limit = st.number_input(
    "Maximum Bias (%)",
    value=2.0
)

# ==========================================================
# MAIN
# ==========================================================

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

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
            .to_numpy()
        )

        response = (
            df["Sensor_Response_umol_mol"]
            .astype(float)
            .to_numpy()
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
            -
            np.min(generated)
        )

        if span > 0:

            max_residual_percent_fs = (
                max_residual
                / span
            ) * 100

        else:

            max_residual_percent_fs = np.nan

        linearity_status = (
            "PASS"
            if r_squared >= r2_limit
            else "FAIL"
        )

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

        # ==================================================
        # PRECISION
        # ==================================================

        precision_rows = []

        for conc, group in df.groupby(
            "Generated_umol_mol"
        ):

            if len(group) > 1:

                mean_response = group[
                    "Sensor_Response_umol_mol"
                ].mean()

                sd_response = group[
                    "Sensor_Response_umol_mol"
                ].std(ddof=1)

                rsd = (
                    sd_response
                    / mean_response
                    * 100
                )

                precision_rows.append({

                    "Generated_umol_mol":
                        conc,

                    "Mean_Response":
                        mean_response,

                    "SD":
                        sd_response,

                    "RSD_percent":
                        rsd

                })

        precision_df = pd.DataFrame(
            precision_rows
        )

        max_rsd = (
            precision_df[
                "RSD_percent"
            ].max()
            if len(precision_df)
            else np.nan
        )

        mean_rsd = (
            precision_df[
                "RSD_percent"
            ].mean()
            if len(precision_df)
            else np.nan
        )

        precision_status = (
            "PASS"
            if (
                pd.notna(max_rsd)
                and max_rsd <= precision_limit
            )
            else "FAIL"
        )

        # ==================================================
        # BIAS
        # ==================================================

        bias_rows = []

        for conc, group in df.groupby(
            "Generated_umol_mol"
        ):

            mean_response = group[
                "Sensor_Response_umol_mol"
            ].mean()

            bias_umol_mol = (
                mean_response
                - conc
            )

            if conc != 0:

                bias_percent = (
                    bias_umol_mol
                    / conc
                    * 100
                )

            else:

                bias_percent = np.nan

            bias_rows.append({

                "Generated_umol_mol":
                    conc,

                "Mean_Response":
                    mean_response,

                "Bias_umol_mol":
                    bias_umol_mol,

                "Bias_percent":
                    bias_percent

            })

        bias_df = pd.DataFrame(
            bias_rows
        )

        max_bias_percent = (
            bias_df[
                "Bias_percent"
            ].abs().max()
        )

        mean_bias_percent = (
            bias_df[
                "Bias_percent"
            ].mean()
        )

        bias_status = (
            "PASS"
            if max_bias_percent <= bias_limit
            else "FAIL"
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
                max_residual
                / span
            ) * 100

        else:

            max_residual_percent_fs = np.nan

        linearity_status = (
            "PASS"
            if r_squared >= r2_limit
            else "FAIL"
        )

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

        # ==================================================
        # PRECISION
        # ==================================================

        precision_rows = []

        for conc, group in df.groupby(
            "Generated_umol_mol"
        ):

            if len(group) > 1:

                mean_response = group[
                    "Sensor_Response_umol_mol"
                ].mean()

                sd_response = group[
                    "Sensor_Response_umol_mol"
                ].std(ddof=1)

                rsd = (
                    sd_response
                    / mean_response
                    * 100
                )

                precision_rows.append({

                    "Generated_umol_mol":
                        conc,

                    "Mean_Response":
                        mean_response,

                    "SD":
                        sd_response,

                    "RSD_percent":
                        rsd

                })

        precision_df = pd.DataFrame(
            precision_rows
        )

        max_rsd = (
            precision_df[
                "RSD_percent"
            ].max()
            if len(precision_df)
            else np.nan
        )

        mean_rsd = (
            precision_df[
                "RSD_percent"
            ].mean()
            if len(precision_df)
            else np.nan
        )

        precision_status = (
            "PASS"
            if (
                pd.notna(max_rsd)
                and max_rsd <= precision_limit
            )
            else "FAIL"
        )

        # ==================================================
        # BIAS
        # ==================================================

        bias_rows = []

        for conc, group in df.groupby(
            "Generated_umol_mol"
        ):

            mean_response = group[
                "Sensor_Response_umol_mol"
            ].mean()

            bias_umol_mol = (
                mean_response
                - conc
            )

            if conc != 0:

                bias_percent = (
                    bias_umol_mol
                    / conc
                    * 100
                )

            else:

                bias_percent = np.nan

            bias_rows.append({

                "Generated_umol_mol":
                    conc,

                "Mean_Response":
                    mean_response,

                "Bias_umol_mol":
                    bias_umol_mol,

                "Bias_percent":
                    bias_percent

            })

        bias_df = pd.DataFrame(
            bias_rows
        )

        max_bias_percent = (
            bias_df[
                "Bias_percent"
            ].abs().max()
        )

        mean_bias_percent = (
            bias_df[
                "Bias_percent"
            ].mean()
        )

        bias_status = (
            "PASS"
            if max_bias_percent <= bias_limit
            else "FAIL"
        )
