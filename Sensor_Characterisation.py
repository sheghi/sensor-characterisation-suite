import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Sensor Characterisation Suite V2",
    layout="wide"
)

st.title("Sensor Characterisation Suite V2")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx", "csv"]
)

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def r3(value):
    try:
        return round(float(value), 3)
    except:
        return value


def load_file(file):

    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    return pd.read_excel(file)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if uploaded:

    df = load_file(uploaded)

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

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

    # ---------------------------------------
    # Convert mA to μmol/mol
    # ---------------------------------------

    df["Measured_umol_mol"] = (
        (df["Signal_mA"] - 4)
        / 16
    ) * 20

    # ---------------------------------------
    # Split ascending / descending
    # ---------------------------------------

    peak_idx = (
        df["Expected_umol_mol"]
        .idxmax()
    )

    ascending = (
        df.iloc[:peak_idx + 1]
        .copy()
    )

    descending = (
        df.iloc[peak_idx:]
        .copy()
    )

    # ---------------------------------------
    # Accuracy
    # ---------------------------------------

    df["Error"] = (
        df["Measured_umol_mol"]
        - df["Expected_umol_mol"]
    )

    df["Error_%"] = (
        df["Error"]
        /
        df["Expected_umol_mol"]
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
            df["Expected_umol_mol"],
            df["Measured_umol_mol"]
        )
    )

    # ---------------------------------------
    # Linearity
    # ---------------------------------------

    X = df[["Expected_umol_mol"]]

    y = df["Measured_umol_mol"]

    model = LinearRegression()

    model.fit(X, y)

    predicted = model.predict(X)

    slope = model.coef_[0]

    intercept = model.intercept_

    r2 = r2_score(
        y,
        predicted
    )

    # ---------------------------------------
    # Resolution
    # ---------------------------------------

    ascending = ascending.sort_values(
        "Expected_umol_mol"
    )

    ascending["Increment"] = (
        ascending["Measured_umol_mol"]
        .diff()
    )

    resolution_mean = (
        ascending["Increment"]
        .dropna()
        .mean()
    )

    resolution_sd = (
        ascending["Increment"]
        .dropna()
        .std()
    )

    resolution_rsd = (
        resolution_sd
        /
        resolution_mean
        * 100
    )

    # ---------------------------------------
    # Reversibility
    # ---------------------------------------

    up = (
        ascending
        .groupby(
            "Expected_umol_mol"
        )[
            "Measured_umol_mol"
        ]
        .mean()
    )

    down = (
        descending
        .groupby(
            "Expected_umol_mol"
        )[
            "Measured_umol_mol"
        ]
        .mean()
    )

    common = (
        up.index
        .intersection(
            down.index
        )
    )

    rev_df = pd.DataFrame({

        "Expected_umol_mol":
        common,

        "Ascending":
        up.loc[common].values,

        "Descending":
        down.loc[common].values

    })

    rev_df["Difference"] = (
        rev_df["Ascending"]
        -
        rev_df["Descending"]
    ).abs()

    max_difference = (
        rev_df["Difference"]
        .max()
    )

    fso = (
        df["Measured_umol_mol"]
        .max()
        -
        df["Measured_umol_mol"]
        .min()
    )

    reversibility_pct = (
        max_difference
        /
        fso
    ) * 100

    # ---------------------------------------
    # Hysteresis
    # ---------------------------------------

    hysteresis_max = (
        rev_df["Difference"]
        .max()
    )

    hysteresis_mean = (
        rev_df["Difference"]
        .mean()
    )

    hysteresis_pct = (
        hysteresis_max
        /
        fso
    ) * 100

    # ---------------------------------------
    # Tabs
    # ---------------------------------------

    tabs = st.tabs([
        "Data",
        "Linearity",
        "Accuracy",
        "Resolution",
        "Reversibility",
        "Hysteresis",
        "Summary",
        "Methods & Formulae"
    ])

    # =======================================
    # DATA
    # =======================================

    with tabs[0.dataframe(
            df.round(3),
            use_container_width=True
        )

    # =======================================
    # LINEARITY
    # =======================================

    with tabsst.metric(
            "Slope",
            r3(slope)
        )

        st.metric(
            "Intercept",
            r3(intercept)
        )

        st.metric(
            "R²",
            r3(r2)
        )

        fig = px.scatter(
            df,
            x="Expected_umol_mol",
            y="Measured_umol_mol",
            title="Linearity Assessment"
        )

        fig.add_scatter(
            x=df["Expected_umol_mol"],
            y=predicted,
            mode="lines",
            name=f"y={slope:.3f}x+{intercept:.3f}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =======================================
    # ACCURACY
    # =======================================

    with tabsst.metric(
            "Mean Error",
            r3(mean_error)
        )

        st.metric(
            "Maximum Error",
            r3(max_error)
        )

        st.metric(
            "RMSE",
            r3(rmse)
        )

        st.dataframe(
            df[
                [
                    "Expected_umol_mol",
                    "Measured_umol_mol",
                    "Error",
                    "Error_%"
                ]
            ].round(3)
        )

    # =======================================
    # RESOLUTION
    # =======================================

    with tabsst.metric(
            "Mean Increment",
            r3(resolution_mean)
        )

        st.metric(
            "SD",
            r3(resolution_sd)
        )

        st.metric(
            "RSD (%)",
            r3(resolution_rsd)
        )

    # =======================================
    # REVERSIBILITY
    # =======================================

    with tabsst.metric(
            "Reversibility (%FSO)",
            r3(reversibility_pct)
        )

        fig_rev = go.Figure()

        fig_rev.add_trace(
            go.Scatter(
                y=df["Measured_umol_mol"],
                mode="lines",
                name="Measured"
            )
        )

        fig_rev.add_trace(
            go.Scatter(
                y=df["Expected_umol_mol"],
                mode="lines",
                name="Expected"
            )
        )

        fig_rev.update_layout(
            title="Reversibility Response"
        )

        st.plotly_chart(
            fig_rev,
            use_container_width=True
        )

    # =======================================
    # HYSTERESIS
    # =======================================

    with tabsst.metric(
            "Maximum Hysteresis",
            r3(hysteresis_max)
        )

        st.metric(
            "Mean Hysteresis",
            r3(hysteresis_mean)
        )

        st.metric(
            "Hysteresis (%FSO)",
            r3(hysteresis_pct)
        )

        fig_loop = go.Figure()

        fig_loop.add_trace(
            go.Scatter(
                x=rev_df["Expected_umol_mol"],
                y=rev_df["Ascending"],
                mode="lines+markers",
                name="Ascending"
            )
        )

        fig_loop.add_trace(
            go.Scatter(
                x=rev_df["Expected_umol_mol"],
                y=rev_df["Descending"],
                mode="lines+markers",
                name="Descending"
            )
        )

        fig_loop.update_layout(
            title="Hysteresis Loop"
        )

        st.plotly_chart(
            fig_loop,
            use_container_width=True
        )

    # =======================================
    # SUMMARY
    # =======================================

    with tabssummary = pd.DataFrame({

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
                r3(slope),
                r3(intercept),
                r3(r2),
                r3(mean_error),
                r3(max_error),
                r3(rmse),
                r3(resolution_mean),
                r3(reversibility_pct),
                r3(hysteresis_pct)
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )

    # =======================================
    # METHODS
    # =======================================

    with tabsst.markdown("""
# Methods & Formulae

## Signal Conversion

Measured concentration (μmol/mol)

((Signal_mA - 4) / 16) × 20

---

## Accuracy

Error = Measured - Expected

Error (%) =
(Error / Expected) × 100

---

## Linearity

Measured = m × Expected + b

Reported:

- Slope
- Intercept
- R²

---

## Resolution

Increment =
Current Measurement − Previous Measurement

RSD (%) =
(SD / Mean) × 100

---

## Reversibility

Reversibility (%FSO) =
Max(|Ascending − Descending|)
÷ FSO × 100

---

## Hysteresis

Hysteresis (%FSO) =
Max(|Ascending − Descending|)
÷ FSO × 100

---

## Full Scale Output

FSO =
Maximum Output − Minimum Output
""")

else:

    st.info(
        "Upload a file containing Expected_umol_mol and Signal_mA."
    )
