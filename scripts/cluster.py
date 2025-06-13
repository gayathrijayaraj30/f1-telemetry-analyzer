import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

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
    return figs

def compare_features_across_drivers(features_df: pd.DataFrame, feature_cols=None, log_scale=False, normalize=False):
    if feature_cols is None:
        feature_cols = ['brake_max', 'throttle_max', 'speed_mean', 'rpm_mean']

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
    return figs  # return list of figures

def pairplot_features(features_df: pd.DataFrame):
    if 'driver_number' not in features_df.columns:
        return None  # or raise an error / message

    numeric_df = features_df.select_dtypes(include=['float64', 'int']).drop(columns=['cluster', 'driver', 'driver_number'], errors='ignore')
    if numeric_df.empty:
        return None

    numeric_df['driver_number'] = features_df['driver_number'].astype(str)
    fig = px.scatter_matrix(numeric_df, dimensions=numeric_df.columns[:-1], color='driver_number', title="Pairplot of Driver Features")
    fig.update_layout(height=800)
    return fig

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.exceptions import NotFittedError

def apply_clustering(scaled_features: pd.DataFrame, original_features: pd.DataFrame, method='kmeans', **kwargs):
    if method == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 3)
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'agglomerative':
        n_clusters = kwargs.get('n_clusters', 3)
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        eps = kwargs.get('eps', 1.0)
        min_samples = kwargs.get('min_samples', 3)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    clusters = model.fit_predict(scaled_features)
    result_df = original_features.copy()
    result_df['cluster'] = clusters

    n_unique = len(set(clusters)) - (1 if -1 in clusters else 0)
    if n_unique > 1:
        try:
            silhouette = silhouette_score(scaled_features, clusters)
        except:
            silhouette = None
        try:
            db_score = davies_bouldin_score(scaled_features, clusters)
        except:
            db_score = None
        try:
            ch_score = calinski_harabasz_score(scaled_features, clusters)
        except:
            ch_score = None
    else:
        silhouette = db_score = ch_score = None

    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score,
        'num_clusters': n_unique
    }

    return result_df, metrics
