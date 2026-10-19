import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==========================================================
# PAGE CONFIGURATION
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

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].apply(
                lambda x: round_sig(x, 4)
            )

    return df


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

    intercept = float(
        model.intercept_
    )

    r2 = float(
