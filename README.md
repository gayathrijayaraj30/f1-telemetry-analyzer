# 🏎️ F1 Telemetry Driver Behavior Clustering & Analysis

**Formula 1 telemetry data analysis and driver behavior clustering using unsupervised learning techniques, with interactive visualization and model explainability.**

🚀 **Live App**: [https://f1-telemetry-analyzer.streamlit.app](https://f1-telemetry-analyzer.streamlit.app)

---

## 📊 Project Overview

This project analyzes Formula 1 telemetry data from race sessions to extract meaningful driver behavior patterns. Using advanced feature engineering and clustering algorithms (KMeans, Agglomerative, DBSCAN), it identifies distinct driving styles based on throttle, braking, speed, and RPM data. 

An interactive [Streamlit](https://streamlit.io) dashboard allows dynamic exploration of driver clusters and telemetry distributions. SHAP explanations enhance interpretability by showing which features influence cluster assignment, offering insights for teams, analysts, and fans.

---

## ✨ Features

- 🔧 **Data Loading**: Automated session data via FastF1.
- 🧠 **Feature Engineering**: Extracts telemetry metrics like max throttle, mean RPM, etc.
- 🔗 **Clustering**: KMeans, Agglomerative, and DBSCAN support.
- 📉 **Dimensionality Reduction**: PCA-based 2D visualization.
- 📈 **Interactive Dashboard**: Built using Streamlit & Plotly.
- 🔍 **Model Explainability**: SHAP explanations for clustering results.
- 🧹 **Robust Filtering**: Filters out noise in DBSCAN and handles incomplete sessions.

---

## 🚀 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/gayathrijayaraj30/f1-telemetry-analyzer.git
   cd f1-telemetry-analyzer
