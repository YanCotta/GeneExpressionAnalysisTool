# Gene Expression Analysis Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/username/GeneExpressionAnalysisTool/workflows/build/badge.svg)](https://github.com/username/GeneExpressionAnalysisTool/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance Python toolkit for comprehensive gene expression analysis, featuring advanced statistical methods and machine learning algorithms.

## 🧬 Overview

This toolkit provides a robust framework for analyzing gene expression data, implementing state-of-the-art bioinformatics methodologies. It's designed for both research and production environments, with emphasis on performance, reliability, and reproducibility.

## ✨ Key Features

### Data Processing Engine
- Intelligent missing value imputation using KNN or MICE algorithms
- Robust outlier detection using MAD (Median Absolute Deviation)
- Automated batch effect correction via ComBat-seq
- Smart low-expression filtering with adaptive thresholds

### Statistical Analysis
- Differential expression analysis using limma-trend methodology
- Multiple testing correction (Benjamini-Hochberg FDR)
- Power analysis and sample size estimation
- Robust normalization (TMM, RLE, or quantile methods)

### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Explained variance analysis
- Feature importance visualization

### Clustering Analysis
- K-means clustering
- Hierarchical clustering with dendrograms
- Silhouette score evaluation

### Visualization
- PCA scatter plots
- Correlation heatmaps
- Expression distribution plots
- Sample quality metrics
- Hierarchical clustering dendrograms

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB RAM minimum (8GB recommended for large datasets)
- CUDA-capable GPU (optional, for accelerated computing)

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install using pip
pip install gene-expression-tool

# Optional: Install GPU support
pip install gene-expression-tool[gpu]
```

### Example Pipeline
```python
from gene_expression_tool import ExpressionAnalysis

# Initialize with advanced configuration
analysis = ExpressionAnalysis(
    normalization_method='tmm',
    batch_correction=True,
    n_jobs=-1  # Use all CPU cores
)

# Load data with automatic format detection
data = analysis.load_data(
    'expression_matrix.csv',
    sample_metadata='metadata.csv',
    verify_integrity=True
)

# Run complete analysis pipeline
results = analysis.run_pipeline(
    min_expression=5,
    fdr_threshold=0.05,
    save_intermediates=True
)
```

## 🔍 Performance Considerations

- Memory usage scales with O(n²) for correlation matrices
- Supports chunked processing for large datasets
- GPU acceleration available for PCA and clustering
- Parallel processing for CPU-intensive operations

## 🛠 Troubleshooting

Common issues and solutions:

1. Memory Errors
```python
# Reduce memory usage
analysis.set_config(use_sparse_matrices=True)
```

2. Performance Issues
```python
# Enable performance optimization
analysis.enable_gpu()
analysis.set_chunk_size(1000)
```

## 🧪 Testing

```bash
# Run test suite
pytest tests/

# Run with coverage
pytest --cov=gene_expression_tool tests/
```

## 📚 Documentation

Detailed documentation is available at [ReadTheDocs](https://gene-expression-tool.readthedocs.io/).

API reference: [API Documentation](https://gene-expression-tool.readthedocs.io/api.html)

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

## 🤝 Contributing

We welcome contributions! 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{gene_expression_tool,
  author = {Cotta, Yan P.},
  title = {Gene Expression Analysis Tool},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YanCotta/GeneExpressionAnalysisTool}
}
```

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
