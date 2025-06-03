# F1 Telemetry Driver Behavior Clustering & Analysis

**Formula 1 telemetry data analysis and driver behavior clustering using unsupervised learning techniques, with interactive visualization and model explainability.**

---

## Project Overview

This project analyzes Formula 1 telemetry data from race sessions to extract meaningful driver behavior patterns. Using advanced feature engineering and clustering algorithms (KMeans, Agglomerative, DBSCAN), it identifies distinct driving styles based on throttle, braking, speed, and RPM data. An interactive Streamlit dashboard allows dynamic exploration of driver clusters and telemetry distributions.

To enhance interpretability, the project integrates SHAP explanations to highlight the features influencing cluster assignments, providing actionable insights for teams, analysts, and enthusiasts.

---

## Features

- **Data Loading:** Automated retrieval of race session telemetry data using FastF1 library.
- **Feature Engineering:** Extraction of driver-specific telemetry features (e.g., max throttle/brake, mean speed/RPM).
- **Clustering:** Supports multiple clustering algorithms with customizable parameters.
- **Dimensionality Reduction:** PCA-based visualization of clusters in 2D space.
- **Interactive Visualization:** Built with Streamlit and Plotly for rich, dynamic exploration.
- **Model Explainability:** SHAP values explain cluster assignment drivers.
- **Robustness:** Filtering of noisy DBSCAN clusters and handling sessions with varying drivers/laps.

---

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/f1-telemetry-clustering.git
   cd f1-telemetry-clustering
