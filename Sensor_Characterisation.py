
        # --------------------------------------------------
        # DATA TAB
        # --------------------------------------------------

        with tab3:

            st.dataframe(
                round_df(linearity_df),
                use_container_width=True
            )

        # --------------------------------------------------
        # METHODS TAB
        # --------------------------------------------------

        with tab4:

            st.markdown(
                """
### Calibration Model

The calibration model is determined by linear regression.

**Sensor Response = m × Generated Amount Fraction + c**

where:

- m = slope
- c = intercept

---

### Coefficient of Determination (R²)

R² is used to assess linearity.

A value closer to 1 indicates better linear agreement between the generated amount fraction and sensor response.

Acceptance criterion:

**R² ≥ User-defined limit**

---

### Residual

Residual:

**Sensor Response − Predicted Response**

---

### Maximum Residual

Maximum Residual:

**max(|Residual|)**

---

### Maximum Residual (%FS)

**(Maximum Absolute Residual ÷ Calibration Span) × 100**

where:

Calibration Span = Maximum Generated Amount Fraction − Minimum Generated Amount Fraction
                """
            )

    except Exception as e:

        st.exception(e)
