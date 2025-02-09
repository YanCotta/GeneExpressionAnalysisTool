from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import silhouette_score
import numpy as np
from typing import Tuple, Dict, Optional
import yaml
from scipy.sparse import issparse
import torch
from torch import nn

class ExpressionModelTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def train_pca(self, data: np.ndarray, 
                use_gpu: bool = False) -> Tuple[PCA, np.ndarray]:
        """Enhanced PCA with GPU support and automatic component selection."""
        if use_gpu and torch.cuda.is_available():
            return self._train_pca_gpu(data)
        
        scaler = RobustScaler()  # More robust to outliers
        scaled_data = scaler.fit_transform(data)
        
        # Optimize number of components
        var_explained = 0
        n_components = 0
        pca_temp = PCA()
        pca_temp.fit(scaled_data)
        
        while var_explained < 0.95 and n_components < min(data.shape):
            n_components += 1
            var_explained = np.sum(pca_temp.explained_variance_ratio_[:n_components])
        
        final_pca = PCA(n_components=n_components)
        transformed_data = final_pca.fit_transform(scaled_data)
        
        return final_pca, transformed_data
    
    def train_clustering(self, data: np.ndarray, 
                        max_clusters: int = 15) -> Dict:
        """Enhanced clustering with multiple algorithms and optimization."""
        results = {}
        
        # K-means with optimization
        kmeans_params = {
            'n_clusters': range(2, max_clusters),
            'init': ['k-means++', 'random'],
            'n_init': [10, 20]
        }
        kmeans = GridSearchCV(
            KMeans(), 
            kmeans_params,
            cv=5,
            scoring='silhouette_score'
        )
        kmeans.fit(data)
        results['kmeans'] = kmeans.best_estimator_
        
        # DBSCAN with automatic eps selection
        distances = np.sort(self._pairwise_distances(data))[:, 1]
        eps_candidates = np.percentile(distances, [10, 15, 20, 25])
        best_score = -1
        best_dbscan = None
        
        for eps in eps_candidates:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(data)
            if len(np.unique(labels)) > 1:  # Valid clustering
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_dbscan = dbscan
        
        results['dbscan'] = best_dbscan
        
        # Spectral clustering for non-linear patterns
        spectral = SpectralClustering(
            n_clusters=kmeans.best_params_['n_clusters'],
            affinity='nearest_neighbors'
        )
        results['spectral'] = spectral.fit(data)
        
        return results
    
    def _train_pca_gpu(self, data: np.ndarray) -> Tuple[PCA, np.ndarray]:
        """GPU-accelerated PCA implementation."""
        data_tensor = torch.from_numpy(data).cuda()
        U, S, V = torch.svd(data_tensor)
        
        # Convert back to CPU for scikit-learn compatibility
        components = V.cpu().numpy().T
        variance = (S.cpu().numpy() ** 2) / (data.shape[0] - 1)
        
        pca = PCA(n_components=data.shape[1])
        pca.components_ = components
        pca.explained_variance_ = variance
        pca.explained_variance_ratio_ = variance / variance.sum()
        
        transformed = data @ components.T
        return pca, transformed
    
    def _pairwise_distances(self, data: np.ndarray) -> np.ndarray:
        """Efficient pairwise distance computation."""
        if issparse(data):
            return self._sparse_pairwise_distances(data)
        return np.sqrt(((data[:, None, :] - data) ** 2).sum(axis=2))
