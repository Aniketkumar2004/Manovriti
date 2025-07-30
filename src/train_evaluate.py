"""
Training and evaluation script for toxic comment classification.
Provides comprehensive model training, evaluation, and visualization capabilities.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve
import logging
from datetime import datetime

# Local imports
from data_preprocessing import (
    TextPreprocessor, FeatureExtractor, load_sample_data, prepare_data
)
from models import (
    LogisticRegressionModel, RandomForestModel, SVMModel, NaiveBayesModel,
    GradientBoostingModel, EnsembleModel, ModelComparison, hyperparameter_tuning
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Comprehensive model training and evaluation class."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.results = {}
        self.models = {}
        self.feature_extractor = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load data from file or use sample data."""
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading data from {data_path}")
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
        else:
            logger.info("Using sample data for demonstration")
            df = load_sample_data()
        
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Toxic samples: {df['toxic'].sum()} ({df['toxic'].mean():.2%})")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, text_column: str = 'comment_text',
                        target_column: str = 'toxic', test_size: float = 0.2) -> Dict:
        """Prepare features for training."""
        logger.info("Preparing features...")
        
        # Split data
        X_train, X_test, y_train, y_test = prepare_data(
            df, text_column, target_column, test_size
        )
        
        # Extract features
        self.feature_extractor = FeatureExtractor()
        X_train_features = self.feature_extractor.extract_tfidf_features(X_train.tolist())
        X_test_features = self.feature_extractor.extract_tfidf_features(X_test.tolist())
        
        logger.info(f"Feature shape: {X_train_features.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_features': X_train_features,
            'X_test_features': X_test_features
        }
    
    def train_models(self, data: Dict, model_configs: Optional[Dict] = None) -> Dict:
        """Train multiple models."""
        logger.info("Starting model training...")
        
        # Default model configurations
        if model_configs is None:
            model_configs = {
                'logistic_regression': {'C': 1.0},
                'random_forest': {'n_estimators': 100, 'max_depth': 10},
                'naive_bayes': {'alpha': 1.0},
                'svm': {'C': 1.0, 'kernel': 'rbf'},
                'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1}
            }
        
        # Initialize models
        models = {
            'logistic_regression': LogisticRegressionModel(**model_configs.get('logistic_regression', {})),
            'random_forest': RandomForestModel(**model_configs.get('random_forest', {})),
            'naive_bayes': NaiveBayesModel(**model_configs.get('naive_bayes', {})),
            'svm': SVMModel(**model_configs.get('svm', {})),
            'gradient_boosting': GradientBoostingModel(**model_configs.get('gradient_boosting', {}))
        }
        
        # Train models
        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                model.train(data['X_train_features'], data['y_train'])
                self.models[name] = model
                
                # Save model
                model_path = os.path.join(self.output_dir, "models", f"{name}.pkl")
                model.save_model(model_path)
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Create ensemble model
        if len(self.models) > 1:
            logger.info("Creating ensemble model...")
            ensemble_models = list(self.models.values())
            ensemble = EnsembleModel(ensemble_models)
            ensemble.train(data['X_train_features'], data['y_train'])
            self.models['ensemble'] = ensemble
            
            # Save ensemble model
            ensemble_path = os.path.join(self.output_dir, "models", "ensemble.pkl")
            with open(ensemble_path, 'wb') as f:
                pickle.dump(ensemble, f)
        
        logger.info(f"Trained {len(self.models)} models successfully")
        return self.models
    
    def evaluate_models(self, data: Dict) -> Dict:
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        comparison = ModelComparison()
        self.results = comparison.compare_models(
            list(self.models.values()),
            data['X_train_features'],
            data['y_train'],
            data['X_test_features'],
            data['y_test']
        )
        
        # Save results
        results_path = os.path.join(self.output_dir, "reports", "model_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create results DataFrame
        results_df = comparison.results_to_dataframe()
        results_csv_path = os.path.join(self.output_dir, "reports", "model_results.csv")
        results_df.to_csv(results_csv_path)
        
        logger.info("Model evaluation completed")
        return self.results
    
    def generate_visualizations(self, data: Dict):
        """Generate comprehensive visualizations."""
        logger.info("Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model comparison plot
        self._plot_model_comparison()
        
        # 2. Confusion matrices
        self._plot_confusion_matrices(data)
        
        # 3. ROC curves
        self._plot_roc_curves(data)
        
        # 4. Feature importance (for tree-based models)
        self._plot_feature_importance()
        
        # 5. Learning curves
        self._plot_learning_curves(data)
        
        # 6. Data distribution
        self._plot_data_distribution(data)
        
        logger.info("Visualizations completed")
    
    def _plot_model_comparison(self):
        """Plot model comparison metrics."""
        if not self.results:
            return
        
        results_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            results_df[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'{metric.capitalize()} Score')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, v in enumerate(results_df[metric]):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "model_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, data: Dict):
        """Plot confusion matrices for all models."""
        n_models = len(self.models)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        for i, (name, model) in enumerate(self.models.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            y_pred = model.predict(data['X_test_features'])
            cm = confusion_matrix(data['y_test'], y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name.replace("_", " ").title()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "confusion_matrices.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, data: Dict):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(data['X_test_features'])
            
            # Handle different probability formats
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_scores = y_pred_proba[:, 1]
            else:
                y_scores = y_pred_proba.flatten()
            
            fpr, tpr, _ = roc_curve(data['y_test'], y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, "plots", "roc_curves.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        tree_models = ['random_forest', 'gradient_boosting']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model.model, 'feature_importances_'):
                    importances = model.model.feature_importances_
                    feature_names = self.feature_extractor.get_feature_names('tfidf')
                    
                    if len(feature_names) == len(importances):
                        # Get top 20 features
                        indices = np.argsort(importances)[::-1][:20]
                        
                        plt.figure(figsize=(12, 8))
                        plt.title(f'Top 20 Feature Importances - {model_name.replace("_", " ").title()}')
                        plt.bar(range(20), importances[indices])
                        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45)
                        plt.tight_layout()
                        
                        plt.savefig(
                            os.path.join(self.output_dir, "plots", f"feature_importance_{model_name}.png"), 
                            dpi=300, bbox_inches='tight'
                        )
                        plt.close()
    
    def _plot_learning_curves(self, data: Dict):
        """Plot learning curves for selected models."""
        selected_models = ['logistic_regression', 'random_forest']
        
        fig, axes = plt.subplots(1, len(selected_models), figsize=(12, 5))
        if len(selected_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(selected_models):
            if model_name in self.models:
                model = self.models[model_name]
                
                train_sizes, train_scores, val_scores = learning_curve(
                    model.model, data['X_train_features'], data['y_train'],
                    train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='f1'
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                ax = axes[i]
                ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
                ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                
                ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
                ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                
                ax.set_title(f'Learning Curve - {model_name.replace("_", " ").title()}')
                ax.set_xlabel('Training Set Size')
                ax.set_ylabel('F1 Score')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "learning_curves.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_distribution(self, data: Dict):
        """Plot data distribution and statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Analysis', fontsize=16, fontweight='bold')
        
        # Class distribution
        class_counts = data['y_train'].value_counts()
        axes[0, 0].pie(class_counts.values, labels=['Non-toxic', 'Toxic'], autopct='%1.1f%%')
        axes[0, 0].set_title('Class Distribution')
        
        # Text length distribution
        text_lengths = data['X_train'].str.len()
        axes[0, 1].hist(text_lengths, bins=50, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].set_xlabel('Character Count')
        axes[0, 1].set_ylabel('Frequency')
        
        # Word count distribution
        word_counts = data['X_train'].str.split().str.len()
        axes[1, 0].hist(word_counts, bins=50, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Word Count Distribution')
        axes[1, 0].set_xlabel('Word Count')
        axes[1, 0].set_ylabel('Frequency')
        
        # Text length by class
        toxic_lengths = text_lengths[data['y_train'] == 1]
        non_toxic_lengths = text_lengths[data['y_train'] == 0]
        
        axes[1, 1].hist(non_toxic_lengths, bins=30, alpha=0.7, label='Non-toxic', color='blue')
        axes[1, 1].hist(toxic_lengths, bins=30, alpha=0.7, label='Toxic', color='red')
        axes[1, 1].set_title('Text Length by Class')
        axes[1, 1].set_xlabel('Character Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "data_distribution.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, data: Dict):
        """Generate comprehensive text report."""
        logger.info("Generating comprehensive report...")
        
        report_path = os.path.join(self.output_dir, "reports", "comprehensive_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Toxic Comment Classification - Comprehensive Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset information
            f.write("## Dataset Information\n\n")
            f.write(f"- **Total samples:** {len(data['y_train']) + len(data['y_test'])}\n")
            f.write(f"- **Training samples:** {len(data['y_train'])}\n")
            f.write(f"- **Test samples:** {len(data['y_test'])}\n")
            f.write(f"- **Toxic samples (train):** {data['y_train'].sum()} ({data['y_train'].mean():.2%})\n")
            f.write(f"- **Toxic samples (test):** {data['y_test'].sum()} ({data['y_test'].mean():.2%})\n\n")
            
            # Model results
            f.write("## Model Performance\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
            f.write("|-------|----------|-----------|--------|----------|----------|\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"| {model_name.replace('_', ' ').title()} | "
                       f"{metrics['accuracy']:.4f} | "
                       f"{metrics['precision']:.4f} | "
                       f"{metrics['recall']:.4f} | "
                       f"{metrics['f1']:.4f} | "
                       f"{metrics['roc_auc']:.4f} |\n")
            
            f.write("\n")
            
            # Best model
            if self.results:
                comparison = ModelComparison()
                comparison.results = self.results
                best_model = comparison.get_best_model('f1')
                f.write(f"**Best performing model (F1-Score):** {best_model.replace('_', ' ').title()}\n\n")
            
            # Detailed classification reports
            f.write("## Detailed Classification Reports\n\n")
            
            for model_name, model in self.models.items():
                f.write(f"### {model_name.replace('_', ' ').title()}\n\n")
                
                y_pred = model.predict(data['X_test_features'])
                report = classification_report(data['y_test'], y_pred)
                
                f.write("```\n")
                f.write(report)
                f.write("\n```\n\n")
            
            # Feature information
            if self.feature_extractor:
                feature_names = self.feature_extractor.get_feature_names('tfidf')
                f.write(f"## Feature Engineering\n\n")
                f.write(f"- **Feature extraction method:** TF-IDF\n")
                f.write(f"- **Number of features:** {len(feature_names)}\n")
                f.write(f"- **N-gram range:** (1, 2)\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Model Selection:** Based on the evaluation metrics, consider using the best performing model for production.\n")
            f.write("2. **Data Quality:** Ensure consistent preprocessing and feature extraction for new data.\n")
            f.write("3. **Monitoring:** Implement continuous monitoring of model performance in production.\n")
            f.write("4. **Improvement:** Consider collecting more diverse training data to improve generalization.\n")
            f.write("5. **Ensemble:** The ensemble model often provides more robust predictions by combining multiple models.\n\n")
            
            # File locations
            f.write("## Generated Files\n\n")
            f.write("- **Models:** `results/models/`\n")
            f.write("- **Visualizations:** `results/plots/`\n")
            f.write("- **Reports:** `results/reports/`\n")
        
        logger.info(f"Comprehensive report saved to {report_path}")
    
    def run_full_pipeline(self, data_path: Optional[str] = None, 
                         model_configs: Optional[Dict] = None):
        """Run the complete training and evaluation pipeline."""
        logger.info("Starting full training and evaluation pipeline...")
        
        # Load data
        df = self.load_data(data_path)
        
        # Prepare features
        data = self.prepare_features(df)
        
        # Train models
        self.train_models(data, model_configs)
        
        # Evaluate models
        self.evaluate_models(data)
        
        # Generate visualizations
        self.generate_visualizations(data)
        
        # Generate report
        self.generate_report(data)
        
        logger.info(f"Pipeline completed! Results saved to {self.output_dir}")
        
        return {
            'models': self.models,
            'results': self.results,
            'feature_extractor': self.feature_extractor
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train and evaluate toxic comment classification models")
    parser.add_argument("--data", type=str, help="Path to data file (CSV or JSON)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to model configuration JSON file")
    
    args = parser.parse_args()
    
    # Load model configurations if provided
    model_configs = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            model_configs = json.load(f)
    
    # Initialize trainer
    trainer = ModelTrainer(args.output)
    
    # Run pipeline
    results = trainer.run_full_pipeline(args.data, model_configs)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Results saved to: {args.output}")
    print("\nModel Performance Summary:")
    
    results_df = pd.DataFrame(results['results']).T
    print(results_df.round(4))


if __name__ == "__main__":
    main()