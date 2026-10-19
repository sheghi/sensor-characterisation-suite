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
# USER INPUTS
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

precision_limit = st.number_input(
    "Maximum Precision (%RSD)",
    min_value=0.0,
    value=2.0,
    step=0.1
)

# ==========================================================
# MAIN
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
            .to_numpy()
        )

        response = (
            df["Sensor_Response_umol_mol"]
            .astype(float)
            .to_numpy()
        )

        # ==================================================
        # LINEARITY
        # ==================================================

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

                if mean_response != 0:

                    rsd_percent = (
                        sd_response
                        /
                        mean_response
                    ) * 100

                else:

                    rsd_percent = np.nan

                precision_rows.append({

                    "Generated_umol_mol":
                        conc,

                    "Mean_Response_umol_mol":
                        mean_response,

                    "SD_umol_mol":
                        sd_response,

                    "RSD_percent":
                        rsd_percent

                })

        precision_df = pd.DataFrame(
            precision_rows
        )

# ===================== END BLOCK 1 =====================
        if len(precision_df) > 0:

            max_rsd = precision_df[
                "RSD_percent"
            ].max()

            mean_rsd = precision_df[
                "RSD_percent"
            ].mean()

        else:

            max_rsd = np.nan
            mean_rsd = np.nan

        precision_status = (
            "PASS"
            if (
                pd.notna(max_rsd)
                and max_rsd <= precision_limit
            )
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
                "Maximum Residual (µmol/mol)",
                "Maximum Residual (%FS)",
                "Maximum RSD (%)",
                "Mean RSD (%)",
                "Linearity Status",
                "Precision Status"

            ],

            "Value": [

                slope,
                intercept,
                r_squared,
                max_residual,
                max_residual_percent_fs,
                max_rsd,
                mean_rsd,
                linearity_status,
                precision_status

            ]

        })

        # ==================================================
        # TABS
        # ==================================================

        tab1, tab2, tab3, tab4 = st.tabs([

            "Summary",
            "Linearity Plot",
            "Linearity Data",
            "Precision"

        ])

        # ==================================================
        # SUMMARY TAB
        # ==================================================

        with tab1:

            st.dataframe(
                round_df(summary_df),
                use_container_width=True
            )

        # ==================================================
        # LINEARITY PLOT
        # ==================================================

        with tab2:

            st.subheader(
                f"Linearity Assessment (R² = {r_squared:.4f})"
            )

            chart_df = pd.DataFrame({

                "Generated":
                    generated,

                "Response":
                    response,

                "Fit":
                    predicted

            })

            points = alt.Chart(
                chart_df
            ).mark_circle(
                size=80,
                color="blue"
            ).encode(

                x=alt.X(
                    "Generated",
                    title="Generated Amount Fraction (µmol/mol)"
                ),

                y=alt.Y(
                    "Response",
                    title="Sensor Response (µmol/mol)"
                )

            )

            line = alt.Chart(

                chart_df.sort_values(
                    "Generated"
                )

            ).mark_line(
                color="red",
                strokeWidth=2
            ).encode(

                x="Generated",

                y="Fit"

            )

            st.altair_chart(
                points + line,
                use_container_width=True
            )

            st.write(
                f"Slope = {slope:.4g}"
            )

            st.write(
                f"Intercept = {intercept:.4g}"
            )

            st.write(
                f"Status = {linearity_status}"
            )

        # ==================================================
        # LINEARITY DATA
        # ==================================================

        with tab3:

            st.dataframe(
                round_df(linearity_df),
                use_container_width=True
            )

        # ==================================================
        # PRECISION
        # ==================================================

        with tab4:

            if len(precision_df) > 0:

                st.dataframe(
                    round_df(precision_df),
                    use_container_width=True
                )

                st.write(
                    f"Maximum RSD (%) = {round_sig(max_rsd)}"
                )

                st.write(
                    f"Mean RSD (%) = {round_sig(mean_rsd)}"
                )

                st.write(
                    f"Status = {precision_status}"
                )

            else:

                st.warning(
                    "No replicate measurements available."
                )

    except Exception as e:

        st.exception(e)
