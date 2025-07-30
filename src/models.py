"""
Machine learning models for toxic comment classification.
Includes traditional ML models, deep learning models, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import joblib
import logging
from abc import ABC, abstractmethod

# Traditional ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

# Deep learning imports (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    from transformers import TrainingArguments, Trainer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Predict probabilities."""
        pass
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Handle binary classification probabilities
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba_binary = y_pred_proba[:, 1]
        else:
            y_pred_proba_binary = y_pred_proba.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_binary)
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.warning("No trained model to save")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for text classification."""
    
    def __init__(self, **kwargs):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            **kwargs
        )
    
    def train(self, X_train, y_train, **kwargs):
        """Train the logistic regression model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} training completed!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest model for text classification."""
    
    def __init__(self, **kwargs):
        super().__init__("Random Forest")
        # Set default values, allow override via kwargs
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        self.model = RandomForestClassifier(**default_params)
    
    def train(self, X_train, y_train, **kwargs):
        """Train the random forest model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} training completed!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)


class SVMModel(BaseModel):
    """Support Vector Machine model for text classification."""
    
    def __init__(self, **kwargs):
        super().__init__("SVM")
        self.model = SVC(
            probability=True,
            random_state=42,
            **kwargs
        )
    
    def train(self, X_train, y_train, **kwargs):
        """Train the SVM model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} training completed!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)


class NaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes model for text classification."""
    
    def __init__(self, **kwargs):
        super().__init__("Naive Bayes")
        self.model = MultinomialNB(**kwargs)
    
    def train(self, X_train, y_train, **kwargs):
        """Train the Naive Bayes model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} training completed!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for text classification."""
    
    def __init__(self, **kwargs):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingClassifier(
            random_state=42,
            **kwargs
        )
    
    def train(self, X_train, y_train, **kwargs):
        """Train the gradient boosting model."""
        logger.info(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} training completed!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)


if TORCH_AVAILABLE:
    class ToxicCommentDataset(Dataset):
        """PyTorch Dataset for toxic comment classification."""
        
        def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
else:
    class ToxicCommentDataset:
        """Placeholder for PyTorch Dataset when torch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for ToxicCommentDataset. Install with: pip install torch")


if TORCH_AVAILABLE:
    class LSTMModel(BaseModel, nn.Module):
        """LSTM model for toxic comment classification."""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128,
                     num_layers: int = 2, dropout: float = 0.3):
            BaseModel.__init__(self, "LSTM")
            nn.Module.__init__(self)
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                               dropout=dropout, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
            self.sigmoid = nn.Sigmoid()
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
    
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, _) = self.lstm(embedded)
            
            # Use the last output from both directions
            output = self.dropout(lstm_out[:, -1, :])
            output = self.fc(output)
            return self.sigmoid(output)
        
        def train_model(self, train_loader, val_loader, epochs: int = 10, lr: float = 0.001):
            """Train the LSTM model."""
            logger.info(f"Training {self.name}...")
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.parameters(), lr=lr)
            
            self.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].float().to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self(input_ids).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self.is_trained = True
            logger.info(f"{self.name} training completed!")
        
        def predict(self, X):
            """Make predictions."""
            self.eval()
            predictions = []
            
            with torch.no_grad():
                for x in X:
                    x_tensor = torch.tensor(x).unsqueeze(0).to(self.device)
                    output = self(x_tensor)
                    pred = (output.cpu().numpy() > 0.5).astype(int)
                    predictions.append(pred[0])
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            """Predict probabilities."""
            self.eval()
            probabilities = []
            
            with torch.no_grad():
                for x in X:
                    x_tensor = torch.tensor(x).unsqueeze(0).to(self.device)
                    output = self(x_tensor)
                    probabilities.append(output.cpu().numpy()[0])
            
            return np.array(probabilities)
else:
    class LSTMModel:
        """Placeholder for LSTM model when torch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMModel. Install with: pip install torch")


# BERT Model (requires transformers and torch)
# Commented out to avoid indentation complexity for basic demo
# if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
#     class BERTModel(BaseModel):
#         """BERT model for toxic comment classification."""
#         
#         def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 2):
#             super().__init__("BERT")
#             self.model_name = model_name
#             self.num_labels = num_labels
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForSequenceClassification.from_pretrained(
#                 model_name, num_labels=num_labels
#             )
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             self.model.to(self.device)

# Placeholder BERT model
class BERTModel:
    """Placeholder for BERT model when transformers/torch is not available."""
    def __init__(self, *args, **kwargs):
        raise ImportError("PyTorch and Transformers are required for BERTModel. Install with: pip install torch transformers")


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple classifiers."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        super().__init__("Ensemble")
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize weights
    
    def train(self, X_train, y_train, **kwargs):
        """Train all models in the ensemble."""
        logger.info(f"Training {self.name} with {len(self.models)} models...")
        
        for model in self.models:
            if not model.is_trained:
                model.train(X_train, y_train, **kwargs)
        
        self.is_trained = True
        logger.info(f"{self.name} training completed!")
    
    def predict(self, X):
        """Make ensemble predictions using majority voting."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted voting
        weighted_predictions = np.zeros(predictions.shape[1])
        for i, weight in enumerate(self.weights):
            weighted_predictions += weight * predictions[i]
        
        return (weighted_predictions > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict ensemble probabilities."""
        probabilities = []
        
        for model in self.models:
            proba = model.predict_proba(X)
            # Handle different probability formats
            if len(proba.shape) > 1 and proba.shape[1] > 1:
                proba = proba[:, 1]  # Take positive class probability
            probabilities.append(proba.flatten())
        
        probabilities = np.array(probabilities)
        
        # Weighted average
        weighted_probabilities = np.zeros(probabilities.shape[1])
        for i, weight in enumerate(self.weights):
            weighted_probabilities += weight * probabilities[i]
        
        # Return in sklearn format (negative class, positive class)
        return np.column_stack([1 - weighted_probabilities, weighted_probabilities])


class ModelComparison:
    """Compare multiple models and their performance."""
    
    def __init__(self):
        self.results = {}
    
    def compare_models(self, models: List[BaseModel], X_train, y_train, X_test, y_test):
        """Compare multiple models and return results."""
        logger.info("Starting model comparison...")
        
        for model in models:
            logger.info(f"Evaluating {model.name}...")
            
            # Train if not already trained
            if not model.is_trained:
                model.train(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            self.results[model.name] = metrics
            
            logger.info(f"{model.name} - F1: {metrics['f1']:.4f}, "
                       f"Accuracy: {metrics['accuracy']:.4f}, "
                       f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return self.results
    
    def get_best_model(self, metric: str = 'f1') -> str:
        """Get the best performing model based on a specific metric."""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0]
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame(self.results).T


def hyperparameter_tuning(model_class, param_grid: Dict, X_train, y_train, cv: int = 5):
    """Perform hyperparameter tuning using GridSearchCV."""
    logger.info(f"Performing hyperparameter tuning for {model_class.__name__}...")
    
    # Create base model
    if model_class == LogisticRegressionModel:
        base_model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_class == RandomForestModel:
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_class == SVMModel:
        base_model = SVC(probability=True, random_state=42)
    else:
        raise ValueError(f"Hyperparameter tuning not implemented for {model_class}")
    
    # Grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_sample_data, prepare_data, FeatureExtractor
    
    # Load and prepare data
    df = load_sample_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Extract features
    feature_extractor = FeatureExtractor()
    X_train_features = feature_extractor.extract_tfidf_features(X_train.tolist())
    X_test_features = feature_extractor.extract_tfidf_features(X_test.tolist())
    
    # Create models
    models = [
        LogisticRegressionModel(),
        RandomForestModel(n_estimators=50),  # Reduced for demo
        NaiveBayesModel(),
        # SVMModel(),  # Commented out for speed in demo
    ]
    
    # Compare models
    comparison = ModelComparison()
    results = comparison.compare_models(models, X_train_features, y_train, X_test_features, y_test)
    
    # Display results
    results_df = comparison.results_to_dataframe()
    print("\nModel Comparison Results:")
    print(results_df.round(4))
    
    best_model = comparison.get_best_model()
    print(f"\nBest model: {best_model}")