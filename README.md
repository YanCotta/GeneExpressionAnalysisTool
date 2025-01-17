# Gene Expression Analysis Tool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional-grade gene expression analysis toolkit implementing robust statistical methods and machine learning algorithms, optimized for high-performance computing environments.

## Project Structure

```
Core/
├── utils.py           # Core statistical and preprocessing utilities
├── model_training.py  # ML model implementations and training logic
├── model_evaluation.py # Evaluation metrics and validation tools
├── main.py           # Pipeline orchestration and CLI interface
└── hyperparams.yaml  # Configuration and hyperparameters
```

## Technical Overview

### Core Components

1. **Statistical Engine** (`utils.py`)
   - TMM/RLE/Quantile normalization
   - Robust outlier detection
   - Quality metrics computation
   - Expression data validation

2. **Model Training** (`model_training.py`)
   - GPU-accelerated PCA implementation
   - Multi-algorithm clustering (K-means, DBSCAN, Spectral)
   - Automatic hyperparameter optimization
   - CUDA support for large-scale computations

3. **Model Evaluation** (`model_evaluation.py`)
   - Bootstrap-based stability assessment
   - GSEA implementation
   - Comprehensive clustering metrics
   - Permutation testing for PCA

4. **Pipeline Orchestration** (`main.py`)
   - Parallel processing support
   - GPU acceleration
   - Progress logging
   - Error handling

## Installation

```bash
git clone https://github.com/username/GeneExpressionAnalysisTool.git
cd GeneExpressionAnalysisTool
pip install -r requirements.txt
```

## Usage Example

```python
from Core.main import GeneExpressionAnalysis

# Initialize with GPU support
analysis = GeneExpressionAnalysis(
    config_path="Core/hyperparams.yaml",
    use_gpu=True
)

# Run analysis pipeline
results = analysis.run_pipeline(
    data_path="expression_data.csv",
    metadata_path="metadata.csv",
    output_dir="results"
)
```

## Configuration

The `hyperparams.yaml` file controls all major parameters:

```yaml
data_processing:
  normalization_method: 'tmm'
  missing_value_threshold: 0.3

pca:
  min_variance_explained: 0.9
  max_components: 50

clustering:
  kmeans:
    max_clusters: 10
    n_init: 10
```

## Changelog

### v3.0.0 (Current)
- Implemented GPU-accelerated PCA
- Added parallel processing support
- Enhanced clustering algorithms
- Introduced GSEA functionality
- Added comprehensive error handling
- Improved documentation and type hints

### v2.0.0 (12/2024)
- ComBat-based batch effect correction
- Comprehensive QC metrics suite
- Hierarchical clustering with dendrograms
- Enhanced visualization pipeline
- Multiple testing correction (FDR)
- Silhouette score analysis for clustering
- Type hints for all functions
- Comprehensive error handling
- Input validation checks
- Improved PCA visualization
- Enhanced documentation
- Optimized data preprocessing
- Updated dependency requirements

## Roadmap

### v3.1.0 (Planned)
- Deep learning integration for expression pattern recognition
- Interactive visualization dashboard
- REST API for remote processing
- Cloud deployment support
- Automated report generation

### v3.2.0 (Future)
- Single-cell RNA-seq analysis support
- Integration with external databases
- Advanced batch effect correction
- Transfer learning capabilities

## Contributing

Feel free to contribute as much as you want!

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{gene_expression_tool,
  author = {Cotta, Yan P.},
  title = {Gene Expression Analysis Tool},
  version = {3.0.0},
  year = {2025},
  url = {https://github.com/YanCotta/GeneExpressionAnalysisTool}
}
```

## Contact

Yan P. Cotta - yanpcotta@gmail.com

---

Built with Python 3.8+, PyTorch, and scikit-learn.