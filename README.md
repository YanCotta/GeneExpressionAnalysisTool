# Gene Expression Analysis Tool 🧬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/gene-expression-analysis-tool/badge/?version=latest)](https://gene-expression-analysis-tool.readthedocs.io)
[![Testing Status](https://github.com/YanCotta/GeneExpressionAnalysisTool/workflows/tests/badge.svg)](https://github.com/YanCotta/GeneExpressionAnalysisTool/actions)

> A professional-grade gene expression analysis toolkit implementing state-of-the-art statistical methods and machine learning algorithms, optimized for high-performance computing environments.

## 📚 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technical Overview](#-technical-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Example](#-usage-example)
- [Configuration](#-configuration)
- [Changelog](#-changelog)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)
- [Contact](#-contact)

## ✨ Features
- **High Performance**: GPU-accelerated computations with CUDA support
- **Robust Statistics**: Implements TMM/RLE/Quantile normalization
- **Advanced ML**: Multi-algorithm clustering with automatic hyperparameter optimization
- **Quality Control**: Comprehensive metrics and validation tools
- **Scalability**: Parallel processing and distributed computing support

## 📁 Project Structure
```
GeneExpressionAnalysisTool/
├── Core/
│   ├── utils.py           # Core statistical and preprocessing utilities
│   ├── model_training.py  # ML model implementations and training logic
│   ├── model_evaluation.py # Evaluation metrics and validation tools
│   ├── main.py           # Pipeline orchestration and CLI interface
│   └── hyperparams.yaml  # Configuration and hyperparameters
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

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/username/GeneExpressionAnalysisTool.git
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start
```python
from Core.main import GeneExpressionAnalysis

# One-line analysis
results = GeneExpressionAnalysis().analyze("data.csv")

# Advanced usage with custom settings
analysis = GeneExpressionAnalysis(
    config_path="Core/hyperparams.yaml",
    use_gpu=True
)
results = analysis.run_pipeline(
    data_path="expression_data.csv",
    metadata_path="metadata.csv",
    output_dir="results"
)
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

## 📋 Changelog

### v3.5.0 (current)
> 🔨 Maintenance Release
- Fixed numerous inconsistencies
- Better integrated project structure and modules
- Updated changelog with v4.0.0 improvements

### v3.0.0 (01/05)
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

## 🗺️ Roadmap

### v4.0.0 (Planned)
> 🎯 Major Release Focus: Complete Framework Overhaul

#### 🏗️ Improve Project Structure
```
GeneExpressionAnalysisTool/
├── tests/                    # Comprehensive test suite
├── examples/                 # Example notebooks
├── docs/                    # Detailed documentation
└── requirements/            # Environment-specific requirements
```

#### Add Essential Bioinformatics Features:
- Batch effect correction
- Multiple testing correction
- Gene set enrichment analysis
- Quality control metrics
- Differential expression analysis

#### Enhance Testing:
- Unit tests for all components
- Integration tests
- Test with real RNA-seq datasets

#### Improve Documentation:
- Add docstrings with biological context
- Create user guide
- Add example workflows
- Document statistical methods

#### Optimize Performance:
- Implement parallel processing
- Add progress tracking
- Optimize memory usage
- Add checkpointing

#### Add Visualization:
- PCA plots
- Heatmaps
- Volcano plots
- Quality control visualizations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

```bibtex
@software{gene_expression_tool,
  author = {Cotta, Yan P.},
  title = {Gene Expression Analysis Tool},
  version = {3.0.0},
  year = {2025},
  url = {https://github.com/YanCotta/GeneExpressionAnalysisTool}
}
```

## 📫 Contact

- **Author**: Yan P. Cotta
- **Email**: yanpcotta@gmail.com
- **GitHub**: [@YanCotta](https://github.com/YanCotta)

---

<div align="center">
  <sub>Built with ❤️ using Python 3.8+, PyTorch, and scikit-learn.</sub>
</div>