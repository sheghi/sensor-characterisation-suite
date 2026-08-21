import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Sensor Characterisation Suite V2",
    layout="wide"
)

st.title("Sensor Characterisation Suite V2")

# --------------------------------------------------
# HELPERS
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
# INPUTS
# --------------------------------------------------

full_scale = st.number_input(
    "Full Scale Concentration (μmol/mol)",
    value=20.0,
    min_value=0.001
)

uploaded = st.file_uploader(
    "Upload Excel or CSV",
    type=["xlsx", "csv"]
)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if uploaded:

    df = load_file(uploaded)

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    required = [
        "Expected_umol_mol",
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

    # --------------------------------------------------
    # CONVERSION
    # --------------------------------------------------

    df["Measured_umol_mol"] = (
        (df["Signal_mA"] - 4)
        / 16
    ) * full_scale

    # --------------------------------------------------
    # ASCENDING / DESCENDING
    # --------------------------------------------------

    peak_idx = df[
        "Expected_umol_mol"
    ].idxmax()

    ascending = (
        df.iloc[:peak_idx + 1]
        .copy()
    )

    descending = (
        df.iloc[peak_idx:]
        .copy()
    )

    # --------------------------------------------------
    # ACCURACY
    # --------------------------------------------------

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

    # --------------------------------------------------
    # LINEARITY
    # --------------------------------------------------

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

    residuals = (
        y - predicted
    )

    # --------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------

    asc_sorted = (
        ascending
        .sort_values(
            "Expected_umol_mol"
        )
    )

    asc_sorted["Increment"] = (
        asc_sorted[
            "Measured_umol_mol"
        ]
        .diff()
    )

    resolution_mean = (
        asc_sorted["Increment"]
        .dropna()
        .mean()
    )

    resolution_sd = (
        asc_sorted["Increment"]
        .dropna()
        .std()
    )

    resolution_rsd = (
        resolution_sd
        / resolution_mean
        * 100
    ) if resolution_mean != 0 else np.nan

    # --------------------------------------------------
    # REVERSIBILITY
    # --------------------------------------------------

    up = (
        ascending
        .groupby(
            "Expected_umol_mol"
        )["Measured_umol_mol"]
        .mean()
    )

    down = (
        descending
        .groupby(
            "Expected_umol_mol"
        )["Measured_umol_mol"]
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
        / fso
        * 100
    )

    # --------------------------------------------------
    # HYSTERESIS
    # --------------------------------------------------

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
        / fso
        * 100
    )

    # --------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------

    min_r2 = 0.995
    max_allowed_error = 0.100
    max_allowed_hysteresis = 5.000
    max_allowed_reversibility = 5.000
    max_allowed_rsd = 10.000

    pass_fail = pd.DataFrame({

        "Metric": [
            "R²",
            "Maximum Error",
            "Resolution RSD (%)",
            "Reversibility (%FSO)",
            "Hysteresis (%FSO)"
        ],

        "Measured": [
            r3(r2),
            r3(max_error),
            r3(resolution_rsd),
            r3(reversibility_pct),
            r3(hysteresis_pct)
        ],

        "Limit": [
            ">= 0.995",
            "<= 0.100",
            "<= 10.000",
            "<= 5.000",
            "<= 5.000"
        ],

        "Result": [
            "PASS" if r2 >= min_r2 else "FAIL",
            "PASS" if max_error <= max_allowed_error else "FAIL",
            "PASS" if resolution_rsd <= max_allowed_rsd else "FAIL",
            "PASS" if reversibility_pct <= max_allowed_reversibility else "FAIL",
            "PASS" if hysteresis_pct <= max_allowed_hysteresis else "FAIL"
        ]
    })

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Data",
        "Linearity",
        "Accuracy",
        "Resolution",
        "Reversibility",
        "Hysteresis",
        "Pass / Fail",
        "Summary",
        "Methods & Formulae"
    ])

    # DATA

    with tab1:

        st.dataframe(
            df.round(3),
            use_container_width=True
        )

    # LINEARITY

    with tab2:

        st.metric("Slope", r3(slope))
        st.metric("Intercept", r3(intercept))
        st.metric("R²", r3(r2))

        fig = px.scatter(
            df,
            x="Expected_umol_mol",
            y="Measured_umol_mol",
            title="Calibration Curve"
        )

        fig.add_scatter(
            x=df["Expected_umol_mol"],
            y=predicted,
            mode="lines",
            name="Regression"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ACCURACY

    with tab3:

        st.metric(
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

    # RESOLUTION

    with tab4:

        st.metric(
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

    # REVERSIBILITY

    with tab5:

        st.metric(
            "Reversibility (%FSO)",
            r3(reversibility_pct)
        )

        fig_rev = go.Figure()

        fig_rev.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Expected_umol_mol"],
                mode="lines+markers",
                name="Expected"
            )
        )

        fig_rev.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Measured_umol_mol"],
                mode="lines+markers",
                name="Measured"
            )
        )

        st.plotly_chart(
            fig_rev,
            use_container_width=True
        )

    # HYSTERESIS

    with tab6:

        st.metric(
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

        st.plotly_chart(
            fig_loop,
            use_container_width=True
        )

        fig_diff = px.bar(
            rev_df,
            x="Expected_umol_mol",
            y="Difference",
            title="Hysteresis Difference"
        )

        st.plotly_chart(
            fig_diff,
            use_container_width=True
        )

    # PASS FAIL

    with tab7:

        st.dataframe(
            pass_fail,
            use_container_width=True
        )

    # SUMMARY

    with tab8:

        summary = pd.DataFrame({

            "Metric": [
                "Slope",
                "Intercept",
                "R²",
                "Mean Error",
                "Maximum Error",
                "MAE",
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
                r3(mae),
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

    # METHODS

    with tab9:

        st.markdown("""
### Signal Conversion

Measured (μmol/mol)

=((Signal_mA - 4) / 16) × Full Scale

---

### Accuracy

Error = Measured − Expected

---

### Linearity

Measured = m × Expected + b

---

### Resolution

RSD (%) = (SD / Mean) × 100

---

### Reversibility

Max(|Ascending − Descending|) ÷ FSO × 100

---

### Hysteresis

Max(|Ascending − Descending|) ÷ FSO × 100
""")

else:

    st.info(
        "Upload a file containing Expected_umol_mol and Signal_mA."
    )
