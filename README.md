# Interactive Clustering Dashboard (Streamlit)

This Streamlit application provides an interactive dashboard for applying and evaluating multiple clustering algorithms on your uploaded dataset. Users can choose clustering techniques, tune hyperparameters, and visualize clusters using PCA.

---

##  Features

-  Upload your own CSV dataset
-  Handles missing data (drop or impute)
-  Select features and clustering algorithms
-  Algorithms included:
  - DBSCAN
  - Mean Shift
  - Gaussian Mixture Model (GMM)
  - Spectral Clustering
  - Hierarchical Clustering
-  Evaluation metrics shown per algorithm:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score
  - Log-likelihood, BIC (for GMM)
  - Adjusted Rand Index, Homogeneity Score (if applicable)
-  PCA-based 2D visualization of cluster outputs
-  Sidebar UI for selecting features, missing value handling, and hyperparameter tuning

---

##  Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

You can install the required dependencies using:

```bash
pip install -r requirements.txt

---

##  Run the app
streamlit run ML_Streamlit.py

How to Use
1. Launch the app.
2. Upload a CSV file containing numeric features.
3. Choose which features to use for clustering in the sidebar.
4. Select how to handle missing values.
5. Choose one or more clustering algorithms.
6. Tune algorithm-specific parameters from the sidebar.
7. View PCA visualizations and evaluation metrics in the main panel.

Output
- Interactive scatterplots with cluster labels (using PCA-reduced data)
- Textual display of clustering evaluation metrics per selected algorithm

Notes
- Only numeric columns will be used for clustering.
- PCA is applied solely for 2D visualization purposes.
- Evaluation metrics depend on cluster validity and may not apply to noise/outliers in DBSCAN or Mean Shift.
