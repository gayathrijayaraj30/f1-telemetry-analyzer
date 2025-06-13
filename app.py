# app.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from scripts.data_loader import get_available_race_data, load_telemetry_data
from scripts.feature_engineer import extract_driver_features, scale_features
from scripts.cluster import apply_clustering
from scripts.visualizer import plot_clusters
from scripts.eda import compare_features_across_drivers, pairplot_features
from scripts.shap_explainer import explain_driver_clusters
import tempfile

st.set_page_config(page_title="F1 Telemetry Analyzer", layout="wide")

st.title("🏎️ F1 Telemetry Dashboard")
st.markdown("Analyze and cluster driver behavior based on telemetry data.")

# Load metadata
years, gps_by_year, sessions_by_year_gp = get_available_race_data()

# Sidebar selections
with st.sidebar:
    st.header("Select Session")
    selected_year = st.selectbox("Year", sorted(years, reverse=True))
    selected_gp = st.selectbox("Grand Prix", gps_by_year[selected_year])
    selected_session = st.selectbox("Session", sessions_by_year_gp[(selected_year, selected_gp)])

    st.header("Clustering Options")
    clustering_method = st.selectbox("Method", ["kmeans", "agglomerative", "dbscan"])
    n_clusters = st.slider("Number of Clusters (for KMeans/Agglomerative)", 2, 10, 3)
    dbscan_eps = st.slider("DBSCAN eps", 0.1, 5.0, 1.0, 0.1)
    dbscan_min_samples = st.slider("DBSCAN min_samples", 1, 10, 3)

    st.header("Visualization Options")
    normalize = st.checkbox("Normalize Features", value=True)
    log_scale = st.checkbox("Log Scale in Plots", value=False)

# Load telemetry data
st.subheader("Loading Telemetry Data")
laps = load_telemetry_data(selected_year, selected_gp, selected_session)
st.success(f"Loaded {len(laps)} laps for session {selected_year} {selected_gp} {selected_session}")

# Extract features
st.subheader("Extracting Driver Features")
features = extract_driver_features(laps)
st.write("Extracted Features", features)

# Race/session summary (after loading laps & features)
st.subheader(f"🏁 Race Summary: {selected_year} {selected_gp} - {selected_session}")

num_laps = len(laps)
num_drivers = laps['Driver'].nunique()

st.markdown(f"""
- **Total Laps Loaded:** {num_laps}
- **Number of Drivers:** {num_drivers}
- **Session:** {selected_session}
""")

avg_speed = features['speed_mean'].mean()
avg_brake = features['brake_mean'].mean()

st.markdown(f"""
- **Average Speed (across drivers):** {avg_speed:.2f} km/h
- **Average Brake Value Recorded (Fastest Lap):** {avg_brake:.2f}
""")

if normalize:
    features_scaled = scale_features(features)
else:
    features_scaled = features.select_dtypes(include="number").copy()

# Apply clustering
st.subheader("🔀 Applying Clustering")
params = {
    "n_clusters": n_clusters,
    "eps": dbscan_eps,
    "min_samples": dbscan_min_samples
}
clustered_df, cluster_metrics = apply_clustering(features_scaled, features, method=clustering_method, **params)
st.write("Clustered Features", clustered_df)

# Compute PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(features_scaled)
clustered_df['pca1'] = pca_components[:, 0]
clustered_df['pca2'] = pca_components[:, 1]

# Visualize clusters (PCA)
st.subheader("PCA Cluster Visualization")
fig = plot_clusters(clustered_df, show=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Distribution Across Drivers")
feature_dist_figs = compare_features_across_drivers(features, normalize=normalize, log_scale=log_scale)
for fig in feature_dist_figs:
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Pairwise Relationships")
pairplot_fig = pairplot_features(features)
if pairplot_fig:
    st.plotly_chart(pairplot_fig, use_container_width=True)
else:
    st.write("Pairplot not available due to missing numeric features.")

# SHAP Explanation
st.subheader("SHAP Cluster Explanation")
with tempfile.TemporaryDirectory() as tmpdir:
    shap_path = os.path.join(tmpdir, "shap_summary.png")
    fig = explain_driver_clusters(clustered_df, save_path=shap_path)
    if fig:
        st.image(shap_path, caption="SHAP Feature Importance", use_column_width=True)
    else:
        st.info("SHAP explanation skipped (only one cluster after filtering or error).")

st.markdown("---")
st.header("Summary & Insights")

num_clusters = clustered_df['cluster'].nunique()
drivers_per_cluster = clustered_df.groupby('cluster')['driver_number'].nunique()

st.write(f"**Number of clusters identified:** {num_clusters}")
st.write("**Number of drivers in each cluster:**")
st.write(drivers_per_cluster)

# Ensure numeric columns are numeric type before aggregation
for col in ['brake_mean', 'throttle_mean', 'speed_mean', 'rpm_mean']:
    clustered_df[col] = pd.to_numeric(clustered_df[col], errors='coerce')

# Aggregate only numeric columns
numeric_cols = ['brake_mean', 'throttle_mean', 'speed_mean', 'rpm_mean']
cluster_summary = clustered_df.groupby('cluster')[numeric_cols].mean()

st.markdown("### Clustering Validation Metrics")

if cluster_metrics['num_clusters'] <= 1:
    st.warning("Not enough clusters were formed to evaluate clustering metrics.")
else:
    silhouette = cluster_metrics['silhouette_score']
    db_score = cluster_metrics['davies_bouldin_score']
    ch_score = cluster_metrics['calinski_harabasz_score']

    if silhouette is not None:
        st.write(f"**Silhouette Score:** {silhouette:.3f}")
    else:
        st.write("**Silhouette Score:** Not available")

    if db_score is not None:
        st.write(f"**Davies–Bouldin Index:** {db_score:.3f}")
    else:
        st.write("**Davies–Bouldin Index:** Not available")

    if ch_score is not None:
        st.write(f"**Calinski–Harabasz Score:** {ch_score:.3f}")
    else:
        st.write("**Calinski–Harabasz Score:** Not available")


st.markdown("### Cluster average feature values:")
st.dataframe(cluster_summary.style.format("{:.2f}"))

st.markdown("""
- Drivers are grouped based on their telemetry behavior such as throttle usage, braking, speed, and RPM patterns.
- Each cluster represents a distinct driving style or pattern identified by the algorithm.
- Features like average braking, throttle application, and speed variation contribute most to these distinctions.
- The PCA plot visualizes how these driving styles differ in a 2D space.
- SHAP analysis highlights which features most influence the cluster assignments.

### Clustering Validation Metrics Explained:
- **Silhouette Score** (range `-1` to `1`): Measures how well each driver fits within its assigned cluster. Values closer to **1** indicate well-separated, dense clusters; values near **0** suggest overlapping clusters.
- **Davies–Bouldin Index** (lower is better): Captures average similarity between clusters. A **lower score** means better cluster separation and compactness.
- **Calinski–Harabasz Score** (higher is better): Reflects how distinct the clusters are based on between- and within-cluster dispersion. **Higher values** indicate more well-defined clusters.

These metrics will automatically update when you change the clustering method or tweak its parameters. Use them together to compare the quality of different clustering models.

### What this means:
- Teams and analysts can use these clusters to identify driver tendencies.
- Insights can support strategy, car setup, and driver coaching by understanding styles.
- This data-driven approach helps uncover subtle behavioral differences not obvious from lap times alone.
""")


st.markdown("---")
