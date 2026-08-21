import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title='Sensor Characterisation Suite', layout='wide')
st.title('Sensor Characterisation Suite')

st.markdown('''
Upload:
- Calibration file containing Expected_ppm
- Sensor file containing Signal_mA and optional Direction (Up/Down)
''')

cal_file = st.file_uploader('Calibration File', type=['xlsx','csv'])
sensor_file = st.file_uploader('Sensor Data File', type=['xlsx','csv'])

def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    return pd.read_excel(file)

if cal_file and sensor_file:
    cal_df = load_file(cal_file)
    sensor_df = load_file(sensor_file)

    cal_df.columns = [str(c).strip() for c in cal_df.columns]
    sensor_df.columns = [str(c).strip() for c in sensor_df.columns]

    if 'Expected_ppm' not in cal_df.columns:
        st.error('Calibration file must contain Expected_ppm')
        st.stop()

    if 'Signal_mA' not in sensor_df.columns:
        st.error('Sensor file must contain Signal_mA')
        st.stop()

    if len(cal_df) != len(sensor_df):
        st.error('Files must contain the same number of rows')
        st.stop()

    df = pd.DataFrame()
    df['Expected_ppm'] = cal_df['Expected_ppm']
    df['Signal_mA'] = sensor_df['Signal_mA']
    df['Direction'] = sensor_df['Direction'] if 'Direction' in sensor_df.columns else 'Up'

    df['Measured_ppm'] = ((df['Signal_mA'] - 4) / 16) * 20

    df['Error'] = df['Measured_ppm'] - df['Expected_ppm']
    df['Error_%'] = (df['Error'] / df['Expected_ppm'].replace(0, np.nan)) * 100

    X = df[['Expected_ppm']]
    y = df['Measured_ppm']

    model = LinearRegression()
    model.fit(X, y)
    predicted = model.predict(X)

    r2 = r2_score(y, predicted)
    slope = model.coef_[0]
    intercept = model.intercept_

    repeatability = df.groupby('Expected_ppm')['Measured_ppm'].agg(['mean', 'std']).reset_index()
    repeatability['RSD_%'] = (repeatability['std'] / repeatability['mean']) * 100

    ordered = df.sort_values('Expected_ppm').reset_index(drop=True)
    ordered['Increment'] = ordered['Measured_ppm'].diff()
    resolution = ordered['Increment'].abs().mean()

    mean_error = df['Error'].mean()
    max_error = df['Error'].abs().max()

    reversibility_pct = np.nan
    hysteresis_pct = np.nan
    hysteresis_df = None

    up = df[df['Direction'].astype(str).str.lower() == 'up'].groupby('Expected_ppm')['Measured_ppm'].mean()
    down = df[df['Direction'].astype(str).str.lower() == 'down'].groupby('Expected_ppm')['Measured_ppm'].mean()

    common = up.index.intersection(down.index)

    if len(common) > 0:
        hysteresis_df = pd.DataFrame({
            'Expected_ppm': common,
            'Ascending': up.loc[common].values,
            'Descending': down.loc[common].values
        })

        hysteresis_df['Difference'] = (hysteresis_df['Ascending'] - hysteresis_df['Descending']).abs()

        max_diff = hysteresis_df['Difference'].max()
        fso = df['Measured_ppm'].max() - df['Measured_ppm'].min()

        reversibility_pct = (max_diff / fso) * 100 if fso else np.nan
        hysteresis_pct = reversibility_pct

    summary = pd.DataFrame({
        'Metric': ['Resolution','Mean Error','Maximum Error','Slope','Intercept','R²','Reversibility (%FSO)','Hysteresis (%FSO)'],
        'Value': [resolution,mean_error,max_error,slope,intercept,r2,reversibility_pct,hysteresis_pct]
    })

    st.subheader('Performance Summary')
    st.dataframe(summary, use_container_width=True)

    st.subheader('Processed Data')
    st.dataframe(df, use_container_width=True)

    fig = px.scatter(df, x='Expected_ppm', y='Measured_ppm', title='Linearity Assessment')
    fig.add_scatter(x=df['Expected_ppm'], y=predicted, mode='lines', name='Best Fit')
    st.plotly_chart(fig, use_container_width=True)

    if hysteresis_df is not None:
        fig_h = px.line(hysteresis_df, x='Expected_ppm', y=['Ascending','Descending'], markers=True, title='Hysteresis')
        st.plotly_chart(fig_h, use_container_width=True)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)

    st.download_button('Download Excel Report', data=buffer.getvalue(), file_name='sensor_characterisation.xlsx')
else:
    st.info('Upload both files to begin analysis.')
