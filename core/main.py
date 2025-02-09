import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import logging
import warnings
import torch
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from .utils import validate_input_data, normalize_expression_data
from .model_training import ExpressionModelTrainer
from .model_evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration settings for analysis pipeline."""
    min_expression: float = 0.1
    n_components: int = 50
    fdr_threshold: float = 0.05
    batch_correction: bool = True
    save_intermediates: bool = False

class GeneExpressionAnalysis:
    """Main class for gene expression analysis pipeline."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        n_jobs: int = -1,
        use_gpu: bool = False
    ):
        """Initialize analysis pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize analysis components with error handling."""
        try:
            self.model_trainer = ExpressionModelTrainer(self.config)
            self.model_evaluator = ModelEvaluator(n_jobs=self.n_jobs)
            
            if self.use_gpu:
                logger.info("GPU acceleration enabled")
                torch.backends.cudnn.benchmark = True
                
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def run_pipeline(
        self,
        data_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Run complete analysis pipeline.
        
        Args:
            data_path: Path to expression data file
            metadata_path: Optional path to metadata file
            output_dir: Optional output directory path
            
        Returns:
            Dict containing analysis results
        """
        try:
            output_dir = Path(output_dir or "results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Data loading and preprocessing
            data = self._load_and_validate_data(data_path)
            
            # Parallel processing pipeline
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = self._run_parallel_analysis(
                    data=data,
                    metadata_path=metadata_path,
                    executor=executor
                )
                
            # Save results
            if output_dir:
                self._save_results(results, output_dir)
                
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

    def _load_and_validate_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """Load and validate input data."""
        try:
            data = pd.read_csv(data_path, index_col='Sample_ID')
            validate_input_data(data)
            return normalize_expression_data(data, method=self.config['normalization_method'])
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _run_parallel_analysis(
        self,
        data: pd.DataFrame,
        metadata_path: Optional[Path],
        executor: ProcessPoolExecutor
    ) -> Dict:
        """Run analysis steps in parallel where possible."""
        results = {}
        
        # Quality control
        logger.info("Performing quality control...")
        results['qc_metrics'] = self._parallel_qc(data, executor)
        
        # Dimensionality reduction
        logger.info("Performing dimensionality reduction...")
        results['pca'] = self.model_trainer.train_pca(
            data.values,
            use_gpu=self.use_gpu
        )
        
        # Clustering
        logger.info("Performing clustering analysis...")
        results['clustering'] = self.model_trainer.train_clustering(
            results['pca'][1]  # Use PCA transformed data
        )
        
        # Differential expression if metadata provided
        if metadata_path:
            logger.info("Performing differential expression analysis...")
            results['de_analysis'] = self._run_de_analysis(
                data,
                metadata_path,
                executor
            )
            
        return results

    def _save_results(self, results: Dict, output_dir: Path) -> None:
        """Save analysis results to files."""
        try:
            for key, data in results.items():
                output_path = output_dir / f"{key}_results.pkl"
                pd.to_pickle(data, output_path)
            logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to save results: {str(e)}")

    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path:
            return yaml.safe_load(Path(config_path).read_text())
        return AnalysisConfig().__dict__

def main():
    """CLI entry point for analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gene Expression Analysis Tool')
    parser.add_argument('--data', required=True, help='Path to expression data')
    parser.add_argument('--metadata', help='Path to metadata file')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    args = parser.parse_args()
    
    try:
        analysis = GeneExpressionAnalysis(
            config_path=args.config,
            use_gpu=args.gpu
        )
        results = analysis.run_pipeline(
            data_path=args.data,
            metadata_path=args.metadata,
            output_dir=args.output
        )
        logger.info("Analysis completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
```
