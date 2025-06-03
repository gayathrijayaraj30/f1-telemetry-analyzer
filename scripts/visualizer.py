import plotly.express as px

def plot_clusters(df, show=True, width=900, height=600):
    fig = px.scatter(
        df,
        x='pca1', y='pca2',
        color='cluster', symbol='driver',
        title="Driver Style Clusters in PCA Space",
        labels={'pca1': 'PCA Component 1', 'pca2': 'PCA Component 2'},
        width=width, height=height
    )
    fig.update_layout(legend_title_text='Cluster / Driver')
    if show:
        fig.show()
    return fig
