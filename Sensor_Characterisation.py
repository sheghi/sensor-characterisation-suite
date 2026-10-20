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

st.title(
    "Sensor Characterisation Suite"
)

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
        sig - int(
            np.floor(
                np.log10(abs(x))
            )
        ) - 1
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

    slope = float(
        model.coef_[0]
    )

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

hysteresis_limit = st.number_input(
    "Maximum Hysteresis (µmol/mol)",
    value=1.0
)

# ==========================================================
# MAIN
# ==========================================================

if uploaded_file is not None:

    try:

        # --------------------------------------------------
        # LOAD FILE
        # --------------------------------------------------

        if uploaded_file.name.endswith(
            ".csv"
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        # --------------------------------------------------
        # VALIDATE INPUT
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
            df[
                "Generated_umol_mol"
            ]
            .astype(float)
            .to_numpy()
        )

        response = (
            df[
                "Sensor_Response_umol_mol"
            ]
            .astype(float)
            .to_numpy()
        )

        # --------------------------------------------------
        # LINEARITY
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
            response
            -
            predicted
        )

        max_residual = np.max(
            np.abs(
                residuals
            )
        )

        span = (
            np.max(
                generated
            )
            -
            np.min(
                generated
            )
        )

        if span > 0:

            max_residual_percent_fs = (
                max_residual
                /
                span
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
                # --------------------------------------------------
        # PRECISION
        # --------------------------------------------------

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
                ].std(
                    ddof=1
                )

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

                    "Mean_Response":
                        mean_response,

                    "SD":
                        sd_response,

                    "RSD_percent":
                        rsd_percent

                })

        precision_df = pd.DataFrame(
            precision_rows
        )

        if len(
            precision_df
        ) > 0:

            max_rsd = (
                precision_df[
                    "RSD_percent"
                ].max()
            )

            mean_rsd = (
                precision_df[
                    "RSD_percent"
                ].mean()
            )

        else:

            max_rsd = np.nan
            mean_rsd = np.nan

        precision_status = (

            "PASS"

            if (
                pd.notna(
                    max_rsd
                )
                and
                max_rsd
                <=
                precision_limit
            )

            else "FAIL"

        )

        # --------------------------------------------------
        # BIAS
        # --------------------------------------------------

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
                ) * 100

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

        if len(
            bias_df
        ) > 0:

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

        else:

            max_bias_percent = np.nan
            mean_bias_percent = np.nan

        bias_status = (

            "PASS"

            if (
                pd.notna(
                    max_bias_percent
                )
                and
                max_bias_percent
                <=
                bias_limit
            )

            else "FAIL"

        )
                # --------------------------------------------------
        # HYSTERESIS
        # Automatic Turnaround Detection
        # --------------------------------------------------

        hysteresis_rows = []

        max_value = np.max(
            generated
        )

        max_index = np.where(
            generated == max_value
        )[0][0]

        up_df = df.iloc[
            :max_index + 1
        ].copy()

        down_df = df.iloc[
            max_index + 1:
        ].copy()

        common_concentrations = sorted(

            set(
                up_df[
                    "Generated_umol_mol"
                ]
            ).intersection(

                set(
                    down_df[
                        "Generated_umol_mol"
                    ]
                )

            )

        )

        for conc in common_concentrations:

            up_response = (

                up_df.loc[
                    up_df[
                        "Generated_umol_mol"
                    ] == conc,
                    "Sensor_Response_umol_mol"
                ]

                .mean()

            )

            down_response = (

                down_df.loc[
                    down_df[
                        "Generated_umol_mol"
                    ] == conc,
                    "Sensor_Response_umol_mol"
                ]

                .mean()

            )

            hysteresis_value = abs(
                up_response
                -
                down_response
            )

            hysteresis_rows.append({

                "Generated_umol_mol":
                    conc,

                "Up_Response":
                    up_response,

                "Down_Response":
                    down_response,

                "Hysteresis_umol_mol":
                    hysteresis_value

            })

        hysteresis_df = pd.DataFrame(
            hysteresis_rows
        )

        if len(
            hysteresis_df
        ) > 0:

            max_hysteresis = float(

                hysteresis_df[
                    "Hysteresis_umol_mol"
                ]

                .abs()

                .max()

            )

        else:

            max_hysteresis = np.nan

        hysteresis_status = (

            "PASS"

            if (
                pd.notna(
                    max_hysteresis
                )
                and
                max_hysteresis
                <=
                hysteresis_limit
            )

            else "FAIL"

        )

        # --------------------------------------------------
        # OVERALL STATUS
        # --------------------------------------------------

        all_statuses = [

            linearity_status,
            precision_status,
            bias_status,
            hysteresis_status

        ]

        overall_status = (

            "PASS"

            if all(
                s == "PASS"
                for s in all_statuses
            )

            else "FAIL"

        )

        # --------------------------------------------------
        # SUMMARY TABLE
        # --------------------------------------------------

        summary_df = pd.DataFrame({

            "Metric": [

                "Slope",
                "Intercept",
                "R²",
                "Maximum Residual (%FS)",
                "Maximum RSD (%)",
                "Mean RSD (%)",
                "Maximum Bias (%)",
                "Mean Bias (%)",
                "Maximum Hysteresis (µmol/mol)",
                "Linearity Status",
                "Precision Status",
                "Bias Status",
                "Hysteresis Status",
                "Overall Status"

            ],

            "Value": [

                slope,
                intercept,
                r_squared,
                max_residual_percent_fs,
                max_rsd,
                mean_rsd,
                max_bias_percent,
                mean_bias_percent,
                max_hysteresis,
                linearity_status,
                precision_status,
                bias_status,
                hysteresis_status,
                overall_status

            ]

        })
                # --------------------------------------------------
        # TABS
        # --------------------------------------------------

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

            "Summary",
            "Linearity Plot",
            "Linearity Data",
            "Precision",
            "Bias",
            "Hysteresis"

        ])

        # --------------------------------------------------
        # SUMMARY TAB
        # --------------------------------------------------

        with tab1:

            st.subheader(
                "Performance Summary"
            )

            st.dataframe(
                round_df(
                    summary_df
                ),
                use_container_width=True
            )

        # --------------------------------------------------
        # LINEARITY PLOT TAB
        # --------------------------------------------------

        with tab2:

            st.subheader(
                f"Linearity Assessment (R² = {r_squared:.4f})"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Slope",
                    f"{slope:.4g}"
                )

            with col2:

                st.metric(
                    "Intercept",
                    f"{intercept:.4g}"
                )

            with col3:

                st.metric(
                    "Status",
                    linearity_status
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
                color="blue",
                size=80
            ).encode(

                x=alt.X(
                    "Generated:Q",
                    title="Generated Amount Fraction (µmol/mol)"
                ),

                y=alt.Y(
                    "Response:Q",
                    title="Sensor Response (µmol/mol)"
                )

            )

            line = alt.Chart(

                chart_df.sort_values(
                    "Generated"
                )

            ).mark_line(
                color="red",
                strokeWidth=3
            ).encode(

                x="Generated:Q",

                y="Fit:Q"

            )

            st.altair_chart(
                points + line,
                use_container_width=True
            )

        # --------------------------------------------------
        # LINEARITY DATA TAB
        # --------------------------------------------------

        with tab3:

            st.subheader(
                "Linearity Data"
            )

            st.dataframe(
                round_df(
                    linearity_df
                ),
                use_container_width=True
            )
                    # --------------------------------------------------
        # PRECISION TAB
        # --------------------------------------------------

        with tab4:

            st.subheader(
                "Precision Assessment"
            )

            if len(precision_df):

                st.dataframe(
                    round_df(
                        precision_df
                    ),
                    use_container_width=True
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Maximum RSD (%)",
                        f"{max_rsd:.4g}"
                    )

                with col2:

                    st.metric(
                        "Mean RSD (%)",
                        f"{mean_rsd:.4g}"
                    )

                with col3:

                    st.metric(
                        "Precision Status",
                        precision_status
                    )

            else:

                st.warning(
                    "No replicate measurements available."
                )

        # --------------------------------------------------
        # BIAS TAB
        # --------------------------------------------------

        with tab5:

            st.subheader(
                "Bias Assessment"
            )

            st.dataframe(
                round_df(
                    bias_df
                ),
                use_container_width=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Maximum Bias (%)",
                    f"{max_bias_percent:.4g}"
                )

            with col2:

                st.metric(
                    "Mean Bias (%)",
                    f"{mean_bias_percent:.4g}"
                )

            with col3:

                st.metric(
                    "Bias Status",
                    bias_status
                )

        # --------------------------------------------------
        # HYSTERESIS TAB
        # --------------------------------------------------

        with tab6:

            st.subheader(
                "Hysteresis Assessment"
            )

            if len(hysteresis_df):

                up_plot_df = up_df.copy()

                up_plot_df[
                    "Direction"
                ] = "Ascending"

                down_plot_df = down_df.copy()

                down_plot_df[
                    "Direction"
                ] = "Descending"

                chart_df = pd.concat(
                    [
                        up_plot_df,
                        down_plot_df
                    ]
                )

                lines = alt.Chart(
                    chart_df
                ).mark_line(
                    strokeWidth=3
                ).encode(

                    x=alt.X(
                        "Generated_umol_mol:Q",
                        title="Generated Amount Fraction (µmol/mol)"
                    ),

                    y=alt.Y(
                        "Sensor_Response_umol_mol:Q",
                        title="Sensor Response (µmol/mol)"
                    ),

                    color=alt.Color(
                        "Direction:N",
                        title="Cycle Direction"
                    ),

                    detail="Direction:N"

                )

                points = alt.Chart(
                    chart_df
                ).mark_circle(
                    size=70
                ).encode(

                    x="Generated_umol_mol:Q",

                    y="Sensor_Response_umol_mol:Q",

                    color="Direction:N"

                )

                st.altair_chart(
                    lines + points,
                    use_container_width=True
                )

                st.dataframe(
                    round_df(
                        hysteresis_df
                    ),
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Maximum Hysteresis (µmol/mol)",
                        f"{max_hysteresis:.4g}"
                    )

                with col2:

                    st.metric(
                        "Hysteresis Status",
                        hysteresis_status
                    )

            else:

                st.warning(
                    "No hysteresis cycle could be identified from the dataset."
                )

    except Exception as e:

        st.exception(e)
