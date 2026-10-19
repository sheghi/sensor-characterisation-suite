# sensor-characterisation-suite

SCOPE

The application evaluates sensor performance using experimentally generated calibration data.

The following performance characteristics are currently assessed:

• Linearity • Precision (Repeatability) • Bias

The software accepts calibration datasets in CSV or Excel format and automatically performs the required calculations.

INPUT DATA REQUIREMENTS

The software requires a dataset containing the following columns:

Generated_umol_mol Sensor_Response_umol_mol

Example:

Generated_umol_mol Sensor_Response_umol_mol

10 10.1 10 10.2 10 10.0

20 20.3 20 20.1 20 20.2

50 50.4 50 50.5 50 50.3

Multiple measurements at the same concentration level are required for precision calculations.

SOURCE OF DATA

Reference Data

Generated concentration values are supplied from calibration systems such as:

• Dynamic dilution systems • Mass flow controller systems • Calibrated gas standards • Certified reference materials

These values are treated as the reference or true concentration.

Sensor Data

Sensor response values are produced by the sensor under test and represent the measured concentration.

LINEARITY ASSESSMENT

Objective

To determine whether the sensor response is proportional to the generated concentration over the calibration range.

Method

Linear regression is applied using:

Sensor Response = (Slope × Generated Concentration) + Intercept

Inputs

• Generated amount fraction (µmol/mol) • Sensor response (µmol/mol)

Outputs

• Slope • Intercept • Predicted response • Residuals • Maximum residual • Maximum residual (% Full Scale) • Coefficient of determination (R²)

Expected Outcome

For an ideal sensor:

Slope ≈ 1

Intercept ≈ 0

R² ≈ 1

Residuals ≈ 0

Typical acceptance criterion:

R² ≥ 0.995

PRECISION ASSESSMENT

Objective

To evaluate repeatability of sensor measurements.

Method

Replicate measurements at each concentration level are grouped.

The following statistics are calculated:

Mean Response

Standard Deviation (SD)

Relative Standard Deviation (RSD%)

RSD (%) = (SD / Mean Response) × 100

Outputs

• Mean response • Standard deviation • RSD (%) • Maximum observed RSD (%) • Mean RSD (%)

Expected Outcome

A highly repeatable sensor should exhibit low variability between replicate measurements.

Typical acceptance criterion:

Maximum RSD ≤ 2.0 %

BIAS ASSESSMENT

Objective

To quantify systematic deviation from the reference value.

Method

Bias = Mean Sensor Response − Generated Concentration

Bias (%) = (Bias / Generated Concentration) × 100

Outputs

• Bias (µmol/mol) • Bias (%) • Maximum Bias (%) • Mean Bias (%)

Expected Outcome

For an unbiased sensor:

Bias ≈ 0

Bias (%) ≈ 0

Typical acceptance criterion:

Maximum Absolute Bias ≤ 2.0 %

GRAPHICAL OUTPUTS

Linearity Plot

The software provides a calibration plot containing:

• Generated concentration on the x-axis • Sensor response on the y-axis • Experimental data points • Linear regression line • R² value

Expected Behaviour

The regression line should closely follow the measured data points for a linear sensor.

REPORTING

All numerical values are reported to four significant figures.

The Summary page shall contain:

• Slope • Intercept • R² • Maximum Residual (%FS) • Maximum RSD (%) • Mean RSD (%) • Maximum Bias (%) • Mean Bias (%) • Linearity Status • Precision Status • Bias Status

PASS / FAIL CRITERIA

Linearity

PASS if:

R² ≥ User-defined R² limit

FAIL if:

R² < User-defined R² limit

Precision

PASS if:

Maximum RSD ≤ User-defined precision limit

FAIL if:

Maximum RSD > User-defined precision limit

Bias

PASS if:

Maximum Absolute Bias ≤ User-defined bias limit

FAIL if:

Maximum Absolute Bias > User-defined bias limit

EXAMPLE TEST DATASET

Generated_umol_mol,Sensor_Response_umol_mol

10,10.0 10,10.1 10,10.2

20,20.1 20,20.2 20,20.3

50,50.3 50,50.4 50,50.5

100,100.5 100,100.6 100,100.7

EXPECTED RESULTS

R² > 0.995

Maximum RSD < 2.0 %

Maximum Bias < 2.0 %

Linearity Status = PASS

Precision Status = PASS

Bias Status = PASS

VERIFICATION REQUIREMENT

The software shall be considered verified when:

Input files are successfully imported.
Calculations match independently verified manual calculations.
Graphs display correctly.
Acceptance criteria are correctly applied.
Appropriate PASS/FAIL outcomes are assigned for all assessed parameters.
Results are reported to four significant figures.

End of Test Plan.
