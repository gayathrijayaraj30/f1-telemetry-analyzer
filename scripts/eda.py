import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

def plot_telemetry_distributions(laps: pd.DataFrame):
    telemetry_data = pd.concat([
        lap.get_telemetry().assign(DriverNumber=lap['DriverNumber'])
        for _, lap in laps.iterrows()
        if lap.get_telemetry() is not None
    ], ignore_index=True)

    figs = []
    for col in ['Throttle', 'Brake', 'Speed', 'RPM']:
        fig = px.histogram(
            telemetry_data,
            x=col,
            color='DriverNumber',
            nbins=50,
            title=f"{col} Distribution by Driver Number",
            marginal="rug"
        )
        figs.append(fig)
    return figs  # Return list of figures

def compare_features_across_drivers(features_df: pd.DataFrame, feature_cols=None, log_scale=False, normalize=False):
    if feature_cols is None:
        feature_cols = ['brake_mean', 'throttle_mean', 'speed_mean', 'rpm_mean']

    df = features_df[['driver_number'] + feature_cols].copy()
    df['driver_number'] = df['driver_number'].astype(str)

    if normalize:
        df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols])

    figs = []
    for col in feature_cols:
        fig = px.box(df, x='driver_number', y=col, title=f"{col.replace('_', ' ').title()} by Driver Number", points="all")
        if log_scale:
            fig.update_yaxes(type="log")
        figs.append(fig)
    return figs


def pairplot_features(features_df: pd.DataFrame):
    if 'driver_number' not in features_df.columns:
        print("❌ 'driver_number' column is missing.")
        return None

    numeric_df = features_df.select_dtypes(include=['float64', 'int']).drop(columns=['cluster', 'driver', 'driver_number'], errors='ignore')
    if numeric_df.empty:
        print("❌ No numeric features available for pairplot.")
        return None

    numeric_df['driver_number'] = features_df['driver_number'].astype(str)
    fig = px.scatter_matrix(
        numeric_df,
        dimensions=numeric_df.columns[:-1],
        color='driver_number',
        title="Pairplot of Driver Features"
    )
    fig.update_layout(height=800)
    return fig  # Return the figure
