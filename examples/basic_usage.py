"""
Basic usage example for toxic comment classification.
Demonstrates core functionality and simple workflows.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import (
    TextPreprocessor, FeatureExtractor, load_sample_data, prepare_data
)
from src.models import (
    LogisticRegressionModel, RandomForestModel, NaiveBayesModel,
    EnsembleModel, ModelComparison
)

def basic_preprocessing_example():
    """Demonstrate basic text preprocessing."""
    print("=" * 50)
    print("TEXT PREPROCESSING EXAMPLE")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Example texts
    examples = [
        "This is a GREAT post!!! Thanks for sharing :)",
        "You're such an idiot... go kill yourself",
        "I disagree with your opinion, but I respect it.",
        "STUPID MORON!!! I HATE YOU SO MUCH!!!"
    ]
    
    print("Original texts and their cleaned versions:\n")
    
    for i, text in enumerate(examples, 1):
        cleaned = preprocessor.preprocess_text(text)
        print(f"Example {i}:")
        print(f"  Original: {text}")
        print(f"  Cleaned:  {cleaned}")
        print()

def single_model_example():
    """Demonstrate training and using a single model."""
    print("=" * 50)
    print("SINGLE MODEL TRAINING EXAMPLE")
    print("=" * 50)
    
    # Load sample data
    print("Loading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} samples")
    print(f"Toxic samples: {df['toxic'].sum()}/{len(df)} ({df['toxic'].mean():.1%})")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    X_train_features = feature_extractor.extract_tfidf_features(X_train.tolist())
    X_test_features = feature_extractor.extract_tfidf_features(X_test.tolist())
    
    print(f"Feature matrix shape: {X_train_features.shape}")
    
    # Train model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegressionModel()
    model.train(X_train_features, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test_features, y_test)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Make predictions on new text
    print("\n" + "=" * 30)
    print("TESTING ON NEW COMMENTS")
    print("=" * 30)
    
    test_comments = [
        "Great job on this project!",
        "You are so stupid and worthless",
        "I don't agree but that's okay",
        "Kill yourself you piece of trash"
    ]
    
    preprocessor = TextPreprocessor()
    
    for comment in test_comments:
        # Preprocess
        cleaned = preprocessor.preprocess_text(comment)
        
        # Extract features
        features = feature_extractor.extract_tfidf_features([cleaned])
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        toxicity_score = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        status = "TOXIC" if prediction else "NON-TOXIC"
        print(f"\nComment: \"{comment}\"")
        print(f"Prediction: {status}")
        print(f"Toxicity Score: {toxicity_score:.3f}")

def model_comparison_example():
    """Demonstrate comparing multiple models."""
    print("=" * 50)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 50)
    
    # Load and prepare data
    df = load_sample_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Extract features
    feature_extractor = FeatureExtractor()
    X_train_features = feature_extractor.extract_tfidf_features(X_train.tolist())
    X_test_features = feature_extractor.extract_tfidf_features(X_test.tolist())
    
    # Create multiple models
    models = [
        LogisticRegressionModel(),
        RandomForestModel(n_estimators=50),  # Reduced for speed
        NaiveBayesModel()
    ]
    
    # Compare models
    print("Training and comparing models...")
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

def ensemble_example():
    """Demonstrate ensemble model usage."""
    print("=" * 50)
    print("ENSEMBLE MODEL EXAMPLE")
    print("=" * 50)
    
    # Load and prepare data
    df = load_sample_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Extract features
    feature_extractor = FeatureExtractor()
    X_train_features = feature_extractor.extract_tfidf_features(X_train.tolist())
    X_test_features = feature_extractor.extract_tfidf_features(X_test.tolist())
    
    # Create individual models
    individual_models = [
        LogisticRegressionModel(),
        RandomForestModel(n_estimators=30),
        NaiveBayesModel()
    ]
    
    # Train individual models
    print("Training individual models...")
    for model in individual_models:
        model.train(X_train_features, y_train)
    
    # Create ensemble
    print("Creating ensemble model...")
    ensemble = EnsembleModel(individual_models, weights=[0.4, 0.4, 0.2])
    ensemble.train(X_train_features, y_train)  # This will skip training since models are already trained
    
    # Compare individual models vs ensemble
    all_models = individual_models + [ensemble]
    comparison = ModelComparison()
    results = comparison.compare_models(
        all_models, X_train_features, y_train, X_test_features, y_test
    )
    
    print("\nIndividual Models vs Ensemble:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{metrics['roc_auc']:<10.4f}")

def batch_prediction_example():
    """Demonstrate batch prediction."""
    print("=" * 50)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Prepare model
    df = load_sample_data()
    X_train, _, y_train, _ = prepare_data(df, test_size=0.1)
    
    feature_extractor = FeatureExtractor()
    X_train_features = feature_extractor.extract_tfidf_features(X_train.tolist())
    
    model = LogisticRegressionModel()
    model.train(X_train_features, y_train)
    
    # Batch of comments to classify
    comments_batch = [
        "Thanks for the helpful information!",
        "You're an absolute moron",
        "I disagree with this approach",
        "Go kill yourself, nobody likes you",
        "Great work on this project!",
        "This is complete garbage",
        "I found this very useful",
        "You make me sick"
    ]
    
    print(f"Classifying {len(comments_batch)} comments...\n")
    
    # Preprocess batch
    preprocessor = TextPreprocessor()
    cleaned_comments = [preprocessor.preprocess_text(comment) for comment in comments_batch]
    
    # Extract features for batch
    batch_features = feature_extractor.extract_tfidf_features(cleaned_comments)
    
    # Batch prediction
    predictions = model.predict(batch_features)
    probabilities = model.predict_proba(batch_features)
    
    # Display results
    print("Batch Prediction Results:")
    print("-" * 70)
    print(f"{'Comment':<35} {'Prediction':<12} {'Score':<8}")
    print("-" * 70)
    
    toxic_count = 0
    for i, (comment, pred, prob) in enumerate(zip(comments_batch, predictions, probabilities)):
        toxicity_score = prob[1] if len(prob) > 1 else prob[0]
        status = "TOXIC" if pred else "NON-TOXIC"
        
        if pred:
            toxic_count += 1
        
        # Truncate long comments for display
        display_comment = comment[:32] + "..." if len(comment) > 35 else comment
        print(f"{display_comment:<35} {status:<12} {toxicity_score:<8.3f}")
    
    print("-" * 70)
    print(f"Summary: {toxic_count}/{len(comments_batch)} comments classified as toxic")

def main():
    """Run all examples."""
    print("🛡️ TOXIC COMMENT CLASSIFICATION - BASIC USAGE EXAMPLES")
    print("=" * 60)
    
    try:
        # Run all examples
        basic_preprocessing_example()
        print("\n")
        
        single_model_example()
        print("\n")
        
        model_comparison_example()
        print("\n")
        
        ensemble_example()
        print("\n")
        
        batch_prediction_example()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()