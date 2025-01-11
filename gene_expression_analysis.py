import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import combat
from goatools import GOEnrichmentStudy
from goatools.obo_parser import GODag

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

def perform_differential_expression(data: pd.DataFrame, group1_ids: List[str], group2_ids: List[str], adj_p_threshold: float = 0.05) -> pd.DataFrame:
    """
    Performs differential expression analysis between two groups using t-test.
    
    Args:
        data (pd.DataFrame): Gene expression data
        group1_ids (List[str]): Sample IDs for group 1
        group2_ids (List[str]): Sample IDs for group 2
        adj_p_threshold (float): Adjusted p-value threshold
    
    Returns:
        pd.DataFrame: Differential expression results
    """
    results = []
    for gene in data.columns:
        group1_expr = data.loc[group1_ids, gene]
        group2_expr = data.loc[group2_ids, gene]
        
        t_stat, p_val = stats.ttest_ind(group1_expr, group2_expr)
        log2fc = np.log2(group1_expr.mean() / group2_expr.mean())
        
        results.append({
            'gene': gene,
            'log2FoldChange': log2fc,
            'pvalue': p_val,
            't_statistic': t_stat
        })
    
    results_df = pd.DataFrame(results)
    # Multiple testing correction
    results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
    return results_df.sort_values('padj')

def correct_batch_effects(data: pd.DataFrame, batch_labels: List[str]) -> pd.DataFrame:
    """
    Corrects for batch effects using ComBat algorithm.
    
    Args:
        data (pd.DataFrame): Gene expression data
        batch_labels (List[str]): Batch information for each sample
    
    Returns:
        pd.DataFrame: Batch-corrected expression data
    """
    # Convert data to appropriate format for ComBat
    combat_data = pd.DataFrame(combat.combat(data.T.values, batch_labels))
    combat_data.index = data.columns
    combat_data.columns = data.index
    return combat_data.T

def perform_hierarchical_clustering(data: pd.DataFrame) -> Dict:
    """
    Performs hierarchical clustering with dendrograms.
    
    Args:
        data (pd.DataFrame): Gene expression data
    
    Returns:
        Dict: Dictionary containing linkage matrices for samples and genes
    """
    # Cluster samples
    sample_linkage = linkage(data, method='ward', metric='euclidean')
    # Cluster genes
    gene_linkage = linkage(data.T, method='ward', metric='euclidean')
    
    return {'sample_linkage': sample_linkage, 'gene_linkage': gene_linkage}

def create_volcano_plot(de_results: pd.DataFrame, fc_threshold: float = 1.0, p_threshold: float = 0.05) -> None:
    """
    Creates a volcano plot from differential expression results.
    
    Args:
        de_results (pd.DataFrame): Differential expression results
        fc_threshold (float): Log2 fold change threshold
        p_threshold (float): Adjusted p-value threshold
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(de_results['log2FoldChange'], -np.log10(de_results['padj']), alpha=0.5)
    
    # Add threshold lines
    plt.axhline(-np.log10(p_threshold), color='r', linestyle='--')
    plt.axvline(-fc_threshold, color='r', linestyle='--')
    plt.axvline(fc_threshold, color='r', linestyle='--')
    
    plt.xlabel('log2 Fold Change')
    plt.ylabel('-log10(adjusted p-value)')
    plt.title('Volcano Plot of Differential Expression')
    plt.show()

def calculate_quality_metrics(data: pd.DataFrame) -> Dict:
    """
    Calculates various quality control metrics for the expression data.
    
    Args:
        data (pd.DataFrame): Gene expression data
    
    Returns:
        Dict: Dictionary containing QC metrics
    """
    metrics = {
        'sample_correlation': data.corr().mean().mean(),
        'genes_detected': (data > 0).sum(axis=0),
        'expression_range': data.max() - data.min(),
        'coefficient_variation': data.std() / data.mean(),
        'missing_values': data.isnull().sum()
    }
    return metrics

def visualize_results(principal_components: np.ndarray, labels: np.ndarray, pca_model: PCA, original_data: pd.DataFrame, qc_metrics: Dict = None) -> None:
    """
    Enhanced visualization function with additional plots.
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 15))
    
    # Original plots
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
    
    # Add QC visualization if metrics are provided
    if qc_metrics:
        ax4 = fig.add_subplot(234)
        sns.boxplot(data=original_data, ax=ax4)
        ax4.set_title('Expression Distribution by Sample')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
        
        ax5 = fig.add_subplot(235)
        sns.histplot(data=qc_metrics['coefficient_variation'], ax=ax5)
        ax5.set_title('Coefficient of Variation Distribution')
        
        ax6 = fig.add_subplot(236)
        sns.heatmap(original_data.corr(), ax=ax6, cmap='coolwarm')
        ax6.set_title('Sample Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Enhanced main function with additional analyses.
    """
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        # Use the full path to your CSV file
        file_path = r"C:\Users\yanpc\OneDrive\Desktop\gene_expression_data.csv"
        df = load_and_preprocess_data(file_path)
        
        # Calculate QC metrics
        print("\nCalculating quality metrics...")
        qc_metrics = calculate_quality_metrics(df)
        
        # Perform batch correction if needed
        # batch_labels = [...] # Add your batch labels here
        # df = correct_batch_effects(df, batch_labels)
        
        # Perform differential expression analysis
        print("\nPerforming differential expression analysis...")
        # Define your groups
        # group1_ids = [...]
        # group2_ids = [...]
        # de_results = perform_differential_expression(df, group1_ids, group2_ids)
        
        # Create volcano plot
        # create_volcano_plot(de_results)
        
        # Perform hierarchical clustering
        print("\nPerforming hierarchical clustering...")
        hierarch_results = perform_hierarchical_clustering(df)
        
        # Perform PCA
        print("\nPerforming PCA...")
        principal_components, pca_model = perform_pca(df)
        
        # Perform clustering
        print("\nPerforming clustering...")
        labels, kmeans_model = cluster_data(principal_components)
        
        # Enhanced visualization with QC metrics
        print("\nGenerating visualizations...")
        visualize_results(principal_components, labels, pca_model, df, qc_metrics)
        
    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")

if __name__ == "__main__":
    main()
