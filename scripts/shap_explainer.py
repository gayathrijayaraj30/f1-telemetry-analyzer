import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
def explain_driver_clusters(features: pd.DataFrame, label_col='cluster', save_path=None):
    try:
        feature_cols = features.select_dtypes(include=['float64', 'int']).columns.drop(label_col)
        
        valid_rows = features[label_col] != -1
        X = features.loc[valid_rows, feature_cols]
        y = features.loc[valid_rows, label_col]

        print("Cluster counts:\n", y.value_counts())

        if y.nunique() <= 1:
            print("Only one cluster found after filtering. Skipping SHAP explanation.")
            return None

        assert not X.isnull().any().any(), "NaNs detected in features"
        assert not X.isin([float('inf'), float('-inf')]).any().any(), "Infinite values detected in features"

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X, y)

        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        fig, _ = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

        return fig
    except Exception as e:
        print(f"Error in SHAP explanation: {e}")
        return None