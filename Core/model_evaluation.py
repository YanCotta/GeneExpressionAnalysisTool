from sklearn.metrics import *
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import pdist, squareform
import gseapy as gp

class ModelEvaluator:
    def __init__(self, n_bootstrap: int = 1000, n_jobs: int = -1):
        self.n_bootstrap = n_bootstrap
        self.n_jobs = n_jobs

    def evaluate_clustering(self, data: np.ndarray, 
                          labels: np.ndarray,
                          metric: str = 'all') -> Dict[str, float]:
        """Enhanced clustering evaluation with bootstrap stability."""
        base_metrics = {
            'silhouette': silhouette_score(data, labels),
            'calinski_harabasz': calinski_harabasz_score(data, labels),
            'davies_bouldin': davies_bouldin_score(data, labels)
        }
        
        # Add stability metrics
        stability_metrics = self._assess_cluster_stability(data, labels)
        separation_metrics = self._calculate_cluster_separation(data, labels)
        
        metrics = {**base_metrics, **stability_metrics, **separation_metrics}
        
        if metric != 'all':
            return {metric: metrics[metric]}
        return metrics

    def _assess_cluster_stability(self, data: np.ndarray, 
                                labels: np.ndarray) -> Dict[str, float]:
        """Assess clustering stability through bootstrapping."""
        stability_scores = []
        
        def bootstrap_iteration(_):
            # Sample with replacement
            indices = np.random.choice(len(data), len(data), replace=True)
            boot_data = data[indices]
            boot_labels = labels[indices]
            return adjusted_rand_score(labels, boot_labels)
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            stability_scores = list(executor.map(bootstrap_iteration, 
                                              range(self.n_bootstrap)))
        
        return {
            'stability_mean': np.mean(stability_scores),
            'stability_std': np.std(stability_scores),
            'stability_ci': np.percentile(stability_scores, [2.5, 97.5])
        }

    def evaluate_differential_expression(self, 
                                      results: pd.DataFrame,
                                      gene_sets: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced differential expression evaluation with GSEA."""
        basic_metrics = {
            'total_deg': len(results[results['padj'] < 0.05]),
            'up_regulated': len(results[(results['padj'] < 0.05) & 
                                     (results['log2FoldChange'] > 0)]),
            'down_regulated': len(results[(results['padj'] < 0.05) & 
                                       (results['log2FoldChange'] < 0)]),
            'effect_size_distribution': results['log2FoldChange'].describe(),
            'significance_distribution': -np.log10(results['padj']).describe()
        }
        
        # Add GSEA if gene sets provided
        if gene_sets:
            gsea_results = self._perform_gsea(results, gene_sets)
            basic_metrics['gsea_results'] = gsea_results
        
        return basic_metrics

    def _perform_gsea(self, results: pd.DataFrame, 
                      gene_sets: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform Gene Set Enrichment Analysis."""
        try:
            # Prepare ranked gene list
            ranked_genes = results.sort_values('log2FoldChange', ascending=False)
            ranked_genes = ranked_genes.index.tolist()
            
            # Run GSEA
            gsea = gp.GSEA(gene_sets=gene_sets)
            gsea.fit(ranked_genes)
            
            return {
                'enriched_pathways': gsea.results,
                'top_pathways': gsea.top_pathways,
                'normalized_es': gsea.normalized_es
            }
        except Exception as e:
            print(f"GSEA analysis failed: {str(e)}")
            return {}

    @staticmethod
    def evaluate_pca(pca_model: Any, data: np.ndarray, 
                     n_permutations: int = 1000) -> Dict[str, Any]:
        """Enhanced PCA evaluation with permutation testing."""
        explained_var = pca_model.explained_variance_ratio_
        
        # Permutation test for component significance
        null_distributions = []
        for _ in range(n_permutations):
            perm_data = np.copy(data)
            for i in range(data.shape[1]):
                np.random.shuffle(perm_data[:, i])
            perm_pca = PCA()
            perm_pca.fit(perm_data)
            null_distributions.append(perm_pca.explained_variance_ratio_)
        
        # Calculate empirical p-values
        null_distributions = np.array(null_distributions)
        p_values = [(null_distributions[:, i] >= explained_var[i]).mean() 
                   for i in range(len(explained_var))]
        
        return {
            'explained_variance_ratio': explained_var,
            'cumulative_variance': np.cumsum(explained_var),
            'n_components': pca_model.n_components_,
            'component_p_values': p_values,
            'significant_components': np.sum(np.array(p_values) < 0.05)
        }
