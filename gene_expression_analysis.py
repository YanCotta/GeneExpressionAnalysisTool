import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path: str, min_expression: float = 0.1) -> pd.DataFrame:
    """
    Loads and preprocesses gene expression data with validation and filtering.
    
    Args:
        file_path (str): Path to the CSV file containing gene expression data
        min_expression (float): Minimum expression value threshold for filtering
    
    Returns:
        pd.DataFrame: Preprocessed gene expression data
    
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        ValueError: If the data format is invalid
    """
    try:
        # Read CSV and set Sample_ID as index
        df = pd.read_csv(file_path, index_col='Sample_ID')
        
        # Validate data format
        if df.empty:
            raise ValueError("Empty dataset detected")
            
        # Remove low-expression genes
        df = df.loc[:, df.mean() > min_expression]
        
        # Log2 transform the data (common in gene expression analysis)
        df = np.log2(df + 1)
        
        # Handle missing values
        if df.isnull().sum().sum() > 0:
            print(f"Warning: Found {df.isnull().sum().sum()} missing values")
            df.fillna(df.mean(), inplace=True)
            
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing data: {str(e)}")

def perform_pca(data: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Performs PCA with advanced preprocessing and explained variance analysis.
    
    Args:
        data (pd.DataFrame): Input gene expression data
        n_components (int): Number of principal components to retain
    
    Returns:
        Tuple[np.ndarray, PCA]: Principal components and PCA model
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Print explained variance information
    explained_var_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio: {explained_var_ratio}")
    print(f"Total variance explained: {sum(explained_var_ratio):.2%}")
    
    return principal_components, pca

def cluster_data(data: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, KMeans]:
    """
    Performs K-means clustering with silhouette score analysis.
    
    Args:
        data (np.ndarray): PCA-transformed data
        n_clusters (int): Number of clusters
    
    Returns:
        Tuple[np.ndarray, KMeans]: Cluster labels and KMeans model
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Calculate silhouette score
    sil_score = silhouette_score(data, labels)
    print(f"Silhouette Score: {sil_score:.3f}")
    
    return labels, kmeans

def visualize_results(principal_components: np.ndarray, labels: np.ndarray, pca_model: PCA, original_data: pd.DataFrame) -> None:
    """
    Creates comprehensive visualizations of the analysis results.
    
    Args:
        principal_components (np.ndarray): PCA-transformed data
        labels (np.ndarray): Cluster labels
        pca_model (PCA): Fitted PCA model
        original_data (pd.DataFrame): Original gene expression data
    """
    # Set up the plotting style
    plt.style.use('seaborn')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. PCA scatter plot with clusters
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(principal_components[:,0], principal_components[:,1], c=labels, cmap='viridis')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Gene Expression PCA and Clustering')
    plt.colorbar(scatter)
    
    # 2. Explained variance plot
    ax2 = fig.add_subplot(132)
    explained_var = pca_model.explained_variance_ratio_
    ax2.plot(range(1, len(explained_var) + 1), np.cumsum(explained_var), 'bo-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Explained Variance Ratio')
    
    # 3. Feature importance heatmap
    ax3 = fig.add_subplot(133)
    top_features = pd.DataFrame(pca_model.components_[:2].T, columns=['PC1', 'PC2'], index=original_data.columns).abs().sum(axis=1).sort_values(ascending=False)[:10]
    sns.heatmap(original_data[top_features.index].corr(), ax=ax3, cmap='coolwarm')
    ax3.set_title('Top Features Correlation')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function demonstrating the complete gene expression analysis pipeline.
    """
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        # Use the full path to your CSV file
        file_path = r"C:\Users\yanpc\OneDrive\Desktop\gene_expression_data.csv"
        df = load_and_preprocess_data(file_path)
        
        # Perform PCA
        print("\nPerforming PCA...")
        principal_components, pca_model = perform_pca(df)
        
        # Perform clustering
        print("\nPerforming clustering...")
        labels, kmeans_model = cluster_data(principal_components)
        
        # Visualize results
        print("\nGenerating visualizations...")
        visualize_results(principal_components, labels, pca_model, df)
        
    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")

if __name__ == "__main__":
    main()
