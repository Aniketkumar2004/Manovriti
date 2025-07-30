"""
Simple example for toxic comment classification using only traditional ML models.
This example avoids PyTorch and Transformers dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import (
    TextPreprocessor, FeatureExtractor, load_sample_data, prepare_data
)
from src.models import (
    LogisticRegressionModel, RandomForestModel, NaiveBayesModel, ModelComparison
)

def main():
    """Run a simple example with traditional ML models."""
    print("🛡️ TOXIC COMMENT CLASSIFICATION - SIMPLE EXAMPLE")
    print("=" * 60)
    
    try:
        # Load sample data
        print("Loading sample data...")
        df = load_sample_data()
        print(f"Loaded {len(df)} samples")
        print(f"Toxic samples: {df['toxic'].sum()}/{len(df)} ({df['toxic'].mean():.1%})")
        
        # Prepare data
        print("\nPreparing data...")
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Extract features
        print("Extracting TF-IDF features...")
        feature_extractor = FeatureExtractor()
        X_train_features = feature_extractor.extract_tfidf_features(X_train.tolist())
        X_test_features = feature_extractor.extract_tfidf_features(X_test.tolist())
        
        print(f"Feature matrix shape: {X_train_features.shape}")
        
        # Create and train models
        print("\nTraining models...")
        models = [
            LogisticRegressionModel(),
            RandomForestModel(n_estimators=50),  # Reduced for speed
            NaiveBayesModel()
        ]
        
        # Train all models
        for model in models:
            print(f"  Training {model.name}...")
            model.train(X_train_features, y_train)
        
        # Compare models
        print("\nEvaluating models...")
        comparison = ModelComparison()
        results = comparison.compare_models(
            models, X_train_features, y_train, X_test_features, y_test
        )
        
        # Display results
        print("\nModel Comparison Results:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<10} {'ROC-AUC':<8}")
        print("-" * 80)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<11.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['roc_auc']:<8.4f}")
        
        # Find best model
        best_model = comparison.get_best_model('f1')
        print(f"\nBest model (by F1-score): {best_model}")
        
        # Test on new comments
        print("\n" + "=" * 40)
        print("TESTING ON NEW COMMENTS")
        print("=" * 40)
        
        test_comments = [
            "Great job on this project!",
            "You are so stupid and worthless",
            "I don't agree but that's okay",
            "This is really helpful, thank you!"
        ]
        
        # Use the best performing model
        best_model_obj = None
        for model in models:
            if model.name == best_model:
                best_model_obj = model
                break
        
        if best_model_obj is None:
            best_model_obj = models[0]  # Fallback to first model
        
        preprocessor = TextPreprocessor()
        
        print(f"Using {best_model_obj.name} for predictions:\n")
        
        for comment in test_comments:
            # Preprocess
            cleaned = preprocessor.preprocess_text(comment)
            
            # Extract features
            features = feature_extractor.extract_tfidf_features([cleaned])
            
            # Predict
            prediction = best_model_obj.predict(features)[0]
            probabilities = best_model_obj.predict_proba(features)[0]
            toxicity_score = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            
            status = "🔴 TOXIC" if prediction else "🟢 NON-TOXIC"
            print(f"Comment: \"{comment}\"")
            print(f"Prediction: {status}")
            print(f"Toxicity Score: {toxicity_score:.3f}")
            print(f"Confidence: {abs(toxicity_score - 0.5) * 2:.3f}")
            print()
        
        print("=" * 60)
        print("✅ Simple example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running simple example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()