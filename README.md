# Gene Expression Analysis Tool

A comprehensive Python tool for analyzing gene expression data with advanced bioinformatics capabilities.

## Biological Context

This code is the first step in analyzing gene expression data:
- Loads your expression matrix
- Filters out noise (low expressing genes)
- Prepares data for downstream analysis (like differential expression or clustering)

It's similar to:
1. Loading your raw RNA-seq counts
2. Removing genes with too few reads
3. Preparing data for DESeq2 or edgeR analysis

## Features

- Data preprocessing and quality control
  - Log2 transformation
  - Missing value handling
  - Low expression filtering
  - Quality metrics calculation
  - Batch effect correction

- Dimensionality Reduction
  - Principal Component Analysis (PCA)
  - Explained variance analysis
  - Feature importance visualization

- Clustering Analysis
  - K-means clustering
  - Hierarchical clustering with dendrograms
  - Silhouette score evaluation

- Differential Expression Analysis
  - T-test based comparison
  - Multiple testing correction (FDR)
  - Volcano plot visualization
  - Fold change analysis

- Visualization
  - PCA scatter plots
  - Correlation heatmaps
  - Expression distribution plots
  - Sample quality metrics
  - Hierarchical clustering dendrograms

## Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn scipy statsmodels combat-py goatools
```

## Usage

### Basic Analysis Pipeline

```python
from gene_expression_analysis import *

# Load and preprocess data
data = load_and_preprocess_data("expression_data.csv")

# Perform PCA
pcs, pca_model = perform_pca(data)

# Cluster data
labels, kmeans = cluster_data(pcs)

# Visualize results
visualize_results(pcs, labels, pca_model, data)
```

### Advanced Features

```python
# Differential Expression Analysis
de_results = perform_differential_expression(data, group1_ids=['sample1', 'sample2'], group2_ids=['sample3', 'sample4'])
create_volcano_plot(de_results)

# Batch Effect Correction
batch_labels = ['batch1', 'batch1', 'batch2', 'batch2']
corrected_data = correct_batch_effects(data, batch_labels)

# Quality Control
qc_metrics = calculate_quality_metrics(data)

# Hierarchical Clustering
hierarch_results = perform_hierarchical_clustering(data)
```

## Data Format

Input CSV file should have the following format:
- First column: Sample_ID (used as index)
- Subsequent columns: Gene expression values
- Header row with gene names

Example:
```
Sample_ID,Gene1,Gene2,Gene3
Sample1,5.2,3.1,4.5
Sample2,6.1,2.8,4.2
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or inquiries, please contact yanpcotta@gmail.com

---

This project demonstrates the intersection of biology and data science, showcasing skills in bioinformatics, machine learning, and data visualization.

# Changelog

## Gene Expression Analysis Tool v2.0
### Added
- ComBat-based batch effect correction
- Comprehensive QC metrics suite
- Hierarchical clustering with dendrograms
- Enhanced visualization pipeline
- Multiple testing correction (FDR)
- Silhouette score analysis for clustering
- Type hints for all functions
- Comprehensive error handling
- Input validation checks

### Changed
- Improved PCA visualization
- Enhanced documentation
- Optimized data preprocessing
- Updated dependency requirements
