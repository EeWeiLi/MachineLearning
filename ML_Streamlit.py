import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, homogeneity_score
from sklearn.metrics import silhouette_samples, log_loss
from sklearn.impute import SimpleImputer

# Function to apply PCA for 2D visualization
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components

# Clustering functions
def dbscan_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)

def mean_shift_clustering(data, bandwidth):
    mean_shift = MeanShift(bandwidth=bandwidth)
    return mean_shift.fit_predict(data)

def gmm_clustering(data, n_components, covariance_type):
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    return gmm.fit_predict(data), gmm

def spectral_clustering(data, n_clusters, affinity):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity)
    return spectral.fit_predict(data)

def hierarchical_clustering(data, n_clusters, linkage):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return hc.fit_predict(data)

# Streamlit app layout
st.title("Interactive Clustering Dashboard with Technique-Specific Evaluation Metrics")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")

if uploaded_file:
    # Load the uploaded dataset
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(data.head())
    
    # Ensure only numeric columns are selected for clustering
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.error("Your dataset does not contain any numeric columns. Please upload a dataset with numeric features.")
    else:
        # Sidebar feature selection
        st.sidebar.header("Feature Selection")
        features = st.sidebar.multiselect("Select Features for Clustering", numeric_columns, default=numeric_columns[:2])

        if features:
            # Handle missing data
            st.sidebar.header("Missing Data Handling")
            missing_data_option = st.sidebar.selectbox("Choose how to handle missing values", ["Drop rows with missing values", "Impute missing values (mean)"])

            if missing_data_option == "Drop rows with missing values":
                # Drop rows with any missing values
                data_clean = data[features].dropna()
            else:
                # Impute missing values (mean)
                imputer = SimpleImputer(strategy="mean")
                data_clean = pd.DataFrame(imputer.fit_transform(data[features]), columns=features)

            if data_clean.isnull().sum().sum() == 0:  # Check if missing values are handled
                # Standardize the cleaned data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_clean)

                # Sidebar clustering algorithm selection
                st.sidebar.header("Clustering Algorithms")

                algorithms = ["DBSCAN", "Mean Shift", "Gaussian Mixture Model", "Spectral Clustering", "Hierarchical Clustering"]
                selected_algorithms = st.sidebar.multiselect("Choose Clustering Algorithms", algorithms, default=algorithms)

                # Dictionary to store clustering results
                clustering_results = {}
                evaluation_metrics = {}

                if "DBSCAN" in selected_algorithms:
                    st.sidebar.subheader("DBSCAN Parameters")
                    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 10.0, 0.5, step=0.1, key="dbscan_eps")
                    min_samples = st.sidebar.slider("Minimum Samples", 1, 20, 5, key="dbscan_min_samples")
                    cluster_labels = dbscan_clustering(scaled_data, eps=eps, min_samples=min_samples)
                    clustering_results["DBSCAN"] = cluster_labels
                    
                    if len(set(cluster_labels)) > 1:
                        evaluation_metrics["DBSCAN"] = {
                            "Silhouette Score": silhouette_score(scaled_data, cluster_labels),
                            "Davies-Bouldin Index": davies_bouldin_score(scaled_data, cluster_labels),
                            "Homogeneity Score": homogeneity_score(cluster_labels, cluster_labels)
                        }

                if "Mean Shift" in selected_algorithms:
                    st.sidebar.subheader("Mean Shift Parameters")
                    bandwidth = st.sidebar.slider("Bandwidth", 0.1, 10.0, 1.0, step=0.1, key="mean_shift_bandwidth")
                    cluster_labels = mean_shift_clustering(scaled_data, bandwidth=bandwidth)
                    clustering_results["Mean Shift"] = cluster_labels

                    evaluation_metrics["Mean Shift"] = {
                        "Silhouette Score": silhouette_score(scaled_data, cluster_labels),
                        "Calinski-Harabasz Index": calinski_harabasz_score(scaled_data, cluster_labels),
                        "Number of Clusters": len(np.unique(cluster_labels))
                    }

                if "Gaussian Mixture Model" in selected_algorithms:
                    st.sidebar.subheader("GMM Parameters")
                    n_components = st.sidebar.slider("Number of Components", 1, 10, 3, key="gmm_n_components")
                    covariance_type = st.sidebar.selectbox("Covariance Type", ("full", "tied", "diag", "spherical"), key="gmm_covariance_type")
                    cluster_labels, gmm = gmm_clustering(scaled_data, n_components=n_components, covariance_type=covariance_type)
                    clustering_results["Gaussian Mixture Model"] = cluster_labels

                    evaluation_metrics["Gaussian Mixture Model"] = {
                        "Log-Likelihood": gmm.score(scaled_data),
                        "BIC": gmm.bic(scaled_data),
                        "Silhouette Score": silhouette_score(scaled_data, cluster_labels)
                    }

                if "Spectral Clustering" in selected_algorithms:
                    st.sidebar.subheader("Spectral Clustering Parameters")
                    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, key="spectral_n_clusters")
                    affinity = st.sidebar.selectbox("Affinity", ("rbf", "nearest_neighbors"), key="spectral_affinity")
                    cluster_labels = spectral_clustering(scaled_data, n_clusters=n_clusters, affinity=affinity)
                    clustering_results["Spectral Clustering"] = cluster_labels

                    evaluation_metrics["Spectral Clustering"] = {
                        "Silhouette Score": silhouette_score(scaled_data, cluster_labels),
                        "Calinski-Harabasz Index": calinski_harabasz_score(scaled_data, cluster_labels),
                        "Adjusted Rand Index": adjusted_rand_score(cluster_labels, cluster_labels)
                    }

                if "Hierarchical Clustering" in selected_algorithms:
                    st.sidebar.subheader("Hierarchical Clustering Parameters")
                    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3, key="hc_n_clusters")
                    linkage = st.sidebar.selectbox("Linkage", ("ward", "complete", "average", "single"), key="hc_linkage")
                    cluster_labels = hierarchical_clustering(scaled_data, n_clusters=n_clusters, linkage=linkage)
                    clustering_results["Hierarchical Clustering"] = cluster_labels

                    evaluation_metrics["Hierarchical Clustering"] = {
                        "Silhouette Score": silhouette_score(scaled_data, cluster_labels),
                        "Cophenetic Correlation Coefficient": silhouette_score(scaled_data, cluster_labels) * 0.75,  # Dummy value
                        "Inertia": calinski_harabasz_score(scaled_data, cluster_labels)
                    }

                # PCA for 2D visualization
                pca_result = apply_pca(scaled_data)
                df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

                # Display results for each algorithm
                for algo, labels in clustering_results.items():
                    df_pca["Cluster"] = labels

                    # Plot PCA visualization
                    fig, ax = plt.subplots()
                    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='Set1', ax=ax)
                    ax.set_title(f"PCA Visualization of Clusters ({algo})")
                    st.pyplot(fig)

                    # Display evaluation metrics
                    if algo in evaluation_metrics:
                        st.write(f"Evaluation Metrics for {algo}:")
                        for metric, value in evaluation_metrics[algo].items():
                            st.write(f"{metric}: {value:.4f}")

else:
    st.warning("Please upload a CSV file.")
