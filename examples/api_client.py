"""
API client example for toxic comment classification.
Demonstrates how to interact with the FastAPI server.
"""

import requests
import json
import time
from typing import List, Dict, Any

class ToxicCommentAPIClient:
    """Client for interacting with the Toxic Comment Classification API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Health check failed: {e}"}
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return [{"error": f"Failed to get models: {e}"}]
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model."""
        try:
            response = self.session.post(f"{self.base_url}/models/{model_name}/load")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to load model {model_name}: {e}"}
    
    def predict_single(self, text: str, model_name: str = None) -> Dict[str, Any]:
        """Predict toxicity for a single comment."""
        try:
            payload = {"text": text}
            if model_name:
                payload["model_name"] = model_name
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Prediction failed: {e}"}
    
    def predict_batch(self, texts: List[str], model_name: str = None) -> Dict[str, Any]:
        """Predict toxicity for multiple comments."""
        try:
            payload = {"texts": texts}
            if model_name:
                payload["model_name"] = model_name
            
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Batch prediction failed: {e}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to get stats: {e}"}
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a specific model."""
        try:
            response = self.session.delete(f"{self.base_url}/models/{model_name}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to unload model {model_name}: {e}"}

def basic_api_example():
    """Demonstrate basic API usage."""
    print("=" * 50)
    print("BASIC API CLIENT EXAMPLE")
    print("=" * 50)
    
    # Initialize client
    client = ToxicCommentAPIClient()
    
    # Health check
    print("1. Checking API health...")
    health = client.health_check()
    if "error" in health:
        print(f"   ❌ {health['error']}")
        return
    else:
        print(f"   ✅ API is {health['status']}")
        print(f"   📊 API Version: {health['api_version']}")
        print(f"   🤖 Loaded models: {health['loaded_models']}")
    
    # Get available models
    print("\n2. Getting available models...")
    models = client.get_models()
    if models and "error" not in models[0]:
        print("   Available models:")
        for model in models:
            status = "✅ Loaded" if model['is_loaded'] else "⏳ Available"
            print(f"   - {model['name']} ({model['type']}) - {status}")
    else:
        print("   ❌ No models available or error occurred")
        return
    
    # Single prediction
    print("\n3. Making single predictions...")
    test_comments = [
        "Great work on this project!",
        "You are such an idiot and I hate you",
        "I disagree with your opinion but respect it",
        "Go kill yourself, nobody likes you"
    ]
    
    for comment in test_comments:
        result = client.predict_single(comment)
        if "error" not in result:
            status = "🔴 TOXIC" if result['is_toxic'] else "🟢 NON-TOXIC"
            print(f"   Comment: \"{comment[:40]}{'...' if len(comment) > 40 else ''}\"")
            print(f"   Result: {status} (Score: {result['toxicity_score']:.3f}, "
                  f"Confidence: {result['confidence']:.3f})")
            print(f"   Model: {result['model_used']}, Time: {result['processing_time_ms']:.1f}ms")
            print()
        else:
            print(f"   ❌ {result['error']}")

def batch_prediction_example():
    """Demonstrate batch prediction."""
    print("=" * 50)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 50)
    
    client = ToxicCommentAPIClient()
    
    # Batch of comments
    comments_batch = [
        "Thanks for sharing this helpful information!",
        "You're absolutely stupid and worthless",
        "I found this article very informative",
        "This is complete garbage, delete it",
        "Great explanation, very clear",
        "You make me sick with your stupidity",
        "I appreciate your hard work on this",
        "I hope you die in a fire"
    ]
    
    print(f"Analyzing {len(comments_batch)} comments in batch...")
    
    result = client.predict_batch(comments_batch)
    
    if "error" not in result:
        predictions = result['predictions']
        total_time = result['processing_time_ms']
        
        print(f"\n✅ Batch processing completed in {total_time:.1f}ms")
        print(f"📊 Average time per comment: {total_time/len(predictions):.1f}ms")
        
        # Summary statistics
        toxic_count = sum(1 for p in predictions if p['is_toxic'])
        avg_toxicity = sum(p['toxicity_score'] for p in predictions) / len(predictions)
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        
        print(f"🔴 Toxic comments: {toxic_count}/{len(predictions)}")
        print(f"📈 Average toxicity score: {avg_toxicity:.3f}")
        print(f"🎯 Average confidence: {avg_confidence:.3f}")
        
        # Detailed results
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'Comment':<45} {'Status':<12} {'Score':<8} {'Confidence':<10}")
        print("-" * 80)
        
        for pred in predictions:
            comment = pred['text']
            status = "TOXIC" if pred['is_toxic'] else "NON-TOXIC"
            score = pred['toxicity_score']
            confidence = pred['confidence']
            
            # Truncate long comments
            display_comment = comment[:42] + "..." if len(comment) > 45 else comment
            print(f"{display_comment:<45} {status:<12} {score:<8.3f} {confidence:<10.3f}")
    
    else:
        print(f"❌ Batch prediction failed: {result['error']}")

def model_management_example():
    """Demonstrate model loading and management."""
    print("=" * 50)
    print("MODEL MANAGEMENT EXAMPLE")
    print("=" * 50)
    
    client = ToxicCommentAPIClient()
    
    # Get current models
    print("1. Current model status:")
    models = client.get_models()
    for model in models:
        status = "Loaded" if model['is_loaded'] else "Available"
        print(f"   - {model['name']}: {status}")
    
    # Try to load a specific model
    if models:
        model_to_load = models[0]['name']
        if not models[0]['is_loaded']:
            print(f"\n2. Loading model: {model_to_load}")
            result = client.load_model(model_to_load)
            if "error" not in result:
                print(f"   ✅ {result['message']}")
            else:
                print(f"   ❌ {result['error']}")
        else:
            print(f"\n2. Model {model_to_load} is already loaded")
    
    # Get usage statistics
    print("\n3. Usage statistics:")
    stats = client.get_stats()
    if "error" not in stats:
        print(f"   📊 Total predictions: {stats['total_predictions']}")
        print(f"   🤖 Loaded models: {stats['loaded_models']}")
        if stats['model_usage_stats']:
            print("   📈 Usage by model:")
            for model, count in stats['model_usage_stats'].items():
                print(f"      - {model}: {count} predictions")
    else:
        print(f"   ❌ {stats['error']}")

def performance_test():
    """Test API performance with multiple requests."""
    print("=" * 50)
    print("PERFORMANCE TEST")
    print("=" * 50)
    
    client = ToxicCommentAPIClient()
    
    # Test single predictions
    test_comment = "This is a test comment for performance testing"
    num_requests = 10
    
    print(f"Testing {num_requests} single predictions...")
    
    start_time = time.time()
    results = []
    
    for i in range(num_requests):
        result = client.predict_single(f"{test_comment} #{i+1}")
        if "error" not in result:
            results.append(result['processing_time_ms'])
        else:
            print(f"   ❌ Request {i+1} failed: {result['error']}")
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    
    if results:
        avg_processing_time = sum(results) / len(results)
        min_time = min(results)
        max_time = max(results)
        
        print(f"\n📊 Performance Results:")
        print(f"   Total requests: {len(results)}/{num_requests}")
        print(f"   Total time: {total_time:.1f}ms")
        print(f"   Average request time: {total_time/len(results):.1f}ms")
        print(f"   Average processing time: {avg_processing_time:.1f}ms")
        print(f"   Min processing time: {min_time:.1f}ms")
        print(f"   Max processing time: {max_time:.1f}ms")
        print(f"   Requests per second: {len(results)/(total_time/1000):.1f}")

def error_handling_example():
    """Demonstrate error handling."""
    print("=" * 50)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 50)
    
    client = ToxicCommentAPIClient()
    
    # Test with invalid text
    print("1. Testing with empty text...")
    result = client.predict_single("")
    if "error" in result:
        print(f"   ✅ Properly handled empty text: {result['error']}")
    else:
        print("   ⚠️ Empty text was processed (unexpected)")
    
    # Test with very long text
    print("\n2. Testing with very long text...")
    long_text = "This is a test comment. " * 1000  # Very long text
    result = client.predict_single(long_text)
    if "error" in result:
        print(f"   ✅ Properly handled long text: {result['error']}")
    else:
        print(f"   ✅ Long text processed successfully (Score: {result.get('toxicity_score', 'N/A')})")
    
    # Test with invalid model name
    print("\n3. Testing with invalid model name...")
    result = client.predict_single("Test comment", model_name="nonexistent_model")
    if "error" in result:
        print(f"   ✅ Properly handled invalid model: {result['error']}")
    else:
        print("   ⚠️ Invalid model name was accepted (unexpected)")
    
    # Test batch with too many comments
    print("\n4. Testing batch with too many comments...")
    many_comments = ["Test comment"] * 150  # Exceeds limit
    result = client.predict_batch(many_comments)
    if "error" in result:
        print(f"   ✅ Properly handled large batch: {result['error']}")
    else:
        print("   ⚠️ Large batch was processed (unexpected)")

def main():
    """Run all API client examples."""
    print("🔗 TOXIC COMMENT CLASSIFICATION - API CLIENT EXAMPLES")
    print("=" * 60)
    
    # Check if API is running
    client = ToxicCommentAPIClient()
    health = client.health_check()
    
    if "error" in health:
        print("❌ API is not running or not accessible!")
        print("Please start the API server first:")
        print("   python src/api.py")
        print("   or")
        print("   uvicorn src.api:app --host 0.0.0.0 --port 8000")
        return
    
    try:
        # Run all examples
        basic_api_example()
        print("\n")
        
        batch_prediction_example()
        print("\n")
        
        model_management_example()
        print("\n")
        
        performance_test()
        print("\n")
        
        error_handling_example()
        
        print("\n" + "=" * 60)
        print("✅ All API client examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running API examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()