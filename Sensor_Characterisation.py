import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="Sensor Characterisation Suite",
    layout="wide"
)

st.title("Sensor Characterisation Suite")

uploaded = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx", "csv"]
)

def load_file(file):

    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    return pd.read_excel(file)

if uploaded:

    df = load_file(uploaded)

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    required = [
        "Expected_ppm",
        "Signal_mA"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        st.error(
            f"Missing columns: {missing}"
        )

        st.stop()

    # -----------------------------------
    # 4–20 mA CONVERSION
    # -----------------------------------

    df["Measured_ppm"] = (
        (df["Signal_mA"] - 4)
        / 16
    ) * 20

    # -----------------------------------
    # ASCENDING / DESCENDING SPLIT
    # -----------------------------------

    peak_idx = (
        df["Expected_ppm"]
        .idxmax()
    )

    ascending = (
        df.iloc[
            :peak_idx + 1
        ]
        .copy()
    )

    descending = (
        df.iloc[
            peak_idx:
        ]
        .copy()
    )

    # -----------------------------------
    # ACCURACY
    # -----------------------------------

    df["Error"] = (
        df["Measured_ppm"]
        - df["Expected_ppm"]
    )

    df["Error_%"] = (
        df["Error"]
        /
        df["Expected_ppm"]
        .replace(0, np.nan)
    ) * 100

    mean_error = (
        df["Error"]
        .mean()
    )

    max_error = (
        df["Error"]
        .abs()
        .max()
    )

    mae = (
        df["Error"]
        .abs()
        .mean()
    )

    rmse = np.sqrt(
        mean_squared_error(
            df["Expected_ppm"],
            df["Measured_ppm"]
        )
    )

    # -----------------------------------
    # LINEARITY
    # -----------------------------------

    X = df[
        ["Expected_ppm"]
    ]

    y = df[
        "Measured_ppm"
    ]

    model = LinearRegression()

    model.fit(X, y)

    predicted = (
        model.predict(X)
    )

    slope = (
        model.coef_[0]
    )

    intercept = (
        model.intercept_
    )

    r2 = r2_score(
        y,
        predicted
    )

    # -----------------------------------
    # RESOLUTION
    # -----------------------------------

    ascending = (
        ascending
        .sort_values(
            "Expected_ppm"
        )
        .reset_index(
            drop=True
        )
    )

    ascending[
        "Increment"
    ] = (
        ascending[
            "Measured_ppm"
        ].diff()
    )

    resolution_mean = (
        ascending[
            "Increment"
        ]
        .dropna()
        .mean()
    )

    resolution_sd = (
        ascending[
            "Increment"
        ]
        .dropna()
        .std()
    )

    resolution_rsd = (
        resolution_sd
        /
        resolution_mean
        * 100
    ) if resolution_mean != 0 else np.nan

    # -----------------------------------
    # REVERSIBILITY
    # -----------------------------------

    up = (
        ascending
        .groupby(
            "Expected_ppm"
        )[
            "Measured_ppm"
        ]
        .mean()
    )

    down = (
        descending
        .groupby(
            "Expected_ppm"
        )[
            "Measured_ppm"
        ]
        .mean()
    )

    common = (
        up.index
        .intersection(
            down.index
        )
    )

    reversibility_df = (
        pd.DataFrame({

            "Expected_ppm":
            common,

            "Ascending":
            up.loc[
                common
            ].values,

            "Descending":
            down.loc[
                common
            ].values

        })
    )

    reversibility_df[
        "Difference"
    ] = (
        reversibility_df[
            "Ascending"
        ]
        -
        reversibility_df[
            "Descending"
        ]
    ).abs()

    max_difference = (
        reversibility_df[
            "Difference"
        ].max()
    )

    fso = (
        df[
            "Measured_ppm"
        ].max()
        -
        df[
            "Measured_ppm"
        ].min()
    )

    reversibility_pct = (
        max_difference
        /
        fso
        * 100
    ) if fso != 0 else np.nan

    # -----------------------------------
    # HYSTERESIS
    # -----------------------------------

    hysteresis_max = (
        reversibility_df[
            "Difference"
        ].max()
    )

    hysteresis_mean = (
        reversibility_df[
            "Difference"
        ].mean()
    )

    hysteresis_pct = (
        hysteresis_max
        /
        fso
        * 100
    ) if fso != 0 else np.nan

    # -----------------------------------
    # TABS
    # -----------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Data",
        "Linearity",
        "Accuracy",
        "Resolution",
        "Reversibility",
        "Hysteresis",
        "Summary"
    ])

    with tab1:

        st.dataframe(
            df,
            use_container_width=True
        )

    with tab2:

        st.metric(
            "Slope",
            round(
                slope,
                6
            )
        )

        st.metric(
            "Intercept",
            round(
                intercept,
                6
            )
        )

        st.metric(
            "R²",
            round(
                r2,
                6
            )
        )

        fig = px.scatter(
            df,
            x="Expected_ppm",
            y="Measured_ppm",
            title="Calibration Curve"
        )

        fig.add_scatter(
            x=df[
                "Expected_ppm"
            ],
            y=predicted,
            mode="lines",
            name="Regression"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab3:

        st.metric(
            "Mean Error",
            round(
                mean_error,
                6
            )
        )

        st.metric(
            "Maximum Error",
            round(
                max_error,
                6
            )
        )

        st.metric(
            "RMSE",
            round(
                rmse,
                6
            )
        )

        st.dataframe(
            df[
                [
                    "Expected_ppm",
                    "Measured_ppm",
                    "Error",
                    "Error_%"
                ]
            ]
        )

    with tab4:

        st.metric(
            "Mean Increment",
            round(
                resolution_mean,
                6
            )
        )

        st.metric(
            "SD",
            round(
                resolution_sd,
                6
            )
        )

        st.metric(
            "RSD (%)",
            round(
                resolution_rsd,
                2
            )
        )

    with tab5:

        st.metric(
            "Reversibility (%FSO)",
            round(
                reversibility_pct,
                4
            )
        )

        fig_rev = px.line(
            reversibility_df,
            x="Expected_ppm",
            y=[
                "Ascending",
                "Descending"
            ],
            markers=True
        )

        st.plotly_chart(
            fig_rev,
            use_container_width=True
        )

    with tab6:

        st.metric(
            "Maximum Hysteresis",
            round(
                hysteresis_max,
                6
            )
        )

        st.metric(
            "Mean Hysteresis",
            round(
                hysteresis_mean,
                6
            )
        )

        st.metric(
            "Hysteresis (%FSO)",
            round(
                hysteresis_pct,
                4
            )
        )

        fig_h = px.bar(
            reversibility_df,
            x="Expected_ppm",
            y="Difference"
        )

        st.plotly_chart(
            fig_h,
            use_container_width=True
        )

    with tab7:

        summary = pd.DataFrame({

            "Metric": [

                "Slope",
                "Intercept",
                "R²",
                "Mean Error",
                "Max Error",
                "RMSE",
                "Resolution",
                "Reversibility (%FSO)",
                "Hysteresis (%FSO)"

            ],

            "Value": [

                slope,
                intercept,
                r2,
                mean_error,
                max_error,
                rmse,
                resolution_mean,
                reversibility_pct,
                hysteresis_pct

            ]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        buffer = BytesIO()

        with pd.ExcelWriter(
            buffer,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                sheet_name="Raw_Data",
                index=False
            )

            reversibility_df.to_excel(
                writer,
                sheet_name="Reversibility",
                index=False
            )

            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )

        st.download_button(
            "Download Excel Report",
            data=buffer.getvalue(),
            file_name="sensor_characterisation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:

    st.info(
        "Upload a file to begin analysis."
    )
