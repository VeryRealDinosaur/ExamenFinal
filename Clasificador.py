import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("sequence_analysis_results.csv")

# Inspect the data
print(data.head())

# Define non-numerical columns
non_numerical_columns = ['Type', 'Accession', 'UniProtID', 'SequenceLength']

# Drop non-numerical columns to get numerical data
numerical_data = data.drop(columns=non_numerical_columns, errors='ignore')

print(numerical_data.dtypes)

# Handle missing values in numerical data
numerical_data = numerical_data.fillna(numerical_data.mean())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Convert back to DataFrame for easier handling
scaled_df = pd.DataFrame(scaled_data, columns=numerical_data.columns)

# Check if scaling worked
print(scaled_df.describe())

# Apply k-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_df)

# Check cluster assignments
print(data[['Type', 'Cluster']].head())

from sklearn.decomposition import PCA

# Reduce to 2 components for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)

# Add PCA results to DataFrame
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis')
plt.title('Clusters Visualized with PCA')
plt.show()

# Check R4X5 cluster
r5x4_cluster = data[data['Type'] == 'R5X4']['Cluster'].values
print(f"R5X4 belongs to Cluster: {r5x4_cluster[0]}")

from sklearn.metrics import silhouette_score

# Silhouette Score
score = silhouette_score(scaled_df, data['Cluster'])
print(f"Silhouette Score: {score}")

# Get cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Convert centroids to DataFrame for comparison
centroids_df = pd.DataFrame(centroids, columns=numerical_data.columns)

print("Cluster Centroids:")
print(centroids_df)
