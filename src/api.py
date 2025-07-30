"""
FastAPI-based REST API for toxic comment classification.
Provides endpoints for model inference and health checks.
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import joblib

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

# Local imports
from data_preprocessing import TextPreprocessor, FeatureExtractor
from models import BaseModel as MLBaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Toxic Comment Classification API",
    description="API for detecting toxic comments using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and preprocessors
loaded_models: Dict[str, MLBaseModel] = {}
text_preprocessor: Optional[TextPreprocessor] = None
feature_extractor: Optional[FeatureExtractor] = None
model_metadata: Dict[str, Any] = {}

# Pydantic models for API
class CommentRequest(BaseModel):
    """Request model for single comment classification."""
    text: str = Field(..., description="The comment text to classify", min_length=1, max_length=5000)
    model_name: Optional[str] = Field(None, description="Specific model to use for prediction")

class BatchCommentRequest(BaseModel):
    """Request model for batch comment classification."""
    texts: List[str] = Field(..., description="List of comment texts to classify", min_items=1, max_items=100)
    model_name: Optional[str] = Field(None, description="Specific model to use for prediction")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    text: str
    is_toxic: bool
    toxicity_score: float
    confidence: float
    model_used: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float

class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    type: str
    is_loaded: bool
    performance_metrics: Optional[Dict[str, float]] = None
    last_used: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    loaded_models: List[str]
    api_version: str

class ModelManager:
    """Manages model loading and caching."""
    
    def __init__(self, models_dir: str = "results/models"):
        self.models_dir = models_dir
        self.models: Dict[str, MLBaseModel] = {}
        self.model_usage_stats: Dict[str, int] = {}
        
    def load_model(self, model_name: str) -> MLBaseModel:
        """Load a specific model."""
        if model_name in self.models:
            return self.models[model_name]
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        
        try:
            # Load the model
            if model_name == "ensemble":
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = joblib.load(model_path)
                # Wrap in our BaseModel if it's a sklearn model
                if not hasattr(model, 'name'):
                    from models import LogisticRegressionModel
                    wrapper = LogisticRegressionModel()
                    wrapper.model = model
                    wrapper.is_trained = True
                    wrapper.name = model_name
                    model = wrapper
            
            self.models[model_name] = model
            self.model_usage_stats[model_name] = 0
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.pkl'):
                models.append(file.replace('.pkl', ''))
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        available_models = self.get_available_models()
        
        return {
            "name": model_name,
            "type": "ensemble" if model_name == "ensemble" else "traditional_ml",
            "is_loaded": model_name in self.models,
            "is_available": model_name in available_models,
            "usage_count": self.model_usage_stats.get(model_name, 0)
        }

# Initialize model manager
model_manager = ModelManager()

# Initialize preprocessors
def initialize_preprocessors():
    """Initialize text preprocessor and feature extractor."""
    global text_preprocessor, feature_extractor
    
    if text_preprocessor is None:
        text_preprocessor = TextPreprocessor()
        logger.info("Text preprocessor initialized")
    
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
        # Try to load pre-fitted feature extractor
        feature_extractor_path = "results/feature_extractor.pkl"
        if os.path.exists(feature_extractor_path):
            try:
                feature_extractor = joblib.load(feature_extractor_path)
                logger.info("Pre-fitted feature extractor loaded")
            except Exception as e:
                logger.warning(f"Could not load pre-fitted feature extractor: {e}")
                feature_extractor = FeatureExtractor()
        logger.info("Feature extractor initialized")

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting Toxic Comment Classification API...")
    initialize_preprocessors()
    
    # Load default model if available
    available_models = model_manager.get_available_models()
    if available_models:
        try:
            default_model = available_models[0]
            model_manager.load_model(default_model)
            logger.info(f"Loaded default model: {default_model}")
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")
    
    logger.info("API startup completed")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Toxic Comment Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        loaded_models=list(model_manager.models.keys()),
        api_version="1.0.0"
    )

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about available models."""
    available_models = model_manager.get_available_models()
    model_info_list = []
    
    for model_name in available_models:
        info = model_manager.get_model_info(model_name)
        model_info_list.append(ModelInfo(
            name=info["name"],
            type=info["type"],
            is_loaded=info["is_loaded"]
        ))
    
    return model_info_list

@app.post("/models/{model_name}/load")
async def load_model(model_name: str, background_tasks: BackgroundTasks):
    """Load a specific model."""
    try:
        model = model_manager.load_model(model_name)
        return {"message": f"Model {model_name} loaded successfully", "model_type": type(model).__name__}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: CommentRequest):
    """Predict toxicity for a single comment."""
    start_time = datetime.now()
    
    # Determine which model to use
    model_name = request.model_name
    if model_name is None:
        available_models = list(model_manager.models.keys())
        if not available_models:
            # Try to load a default model
            all_models = model_manager.get_available_models()
            if all_models:
                model_name = all_models[0]
                model_manager.load_model(model_name)
            else:
                raise HTTPException(status_code=503, detail="No models available")
        else:
            model_name = available_models[0]
    
    # Load model if not already loaded
    if model_name not in model_manager.models:
        try:
            model_manager.load_model(model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    model = model_manager.models[model_name]
    
    try:
        # Preprocess text
        cleaned_text = text_preprocessor.preprocess_text(request.text)
        
        # Extract features
        features = feature_extractor.extract_tfidf_features([cleaned_text])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Handle different probability formats
        if len(probabilities.shape) > 0 and len(probabilities) > 1:
            toxicity_score = probabilities[1]  # Probability of toxic class
        else:
            toxicity_score = probabilities if probabilities.ndim == 0 else probabilities[0]
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(toxicity_score - 0.5) * 2
        
        # Update usage stats
        model_manager.model_usage_stats[model_name] += 1
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            text=request.text,
            is_toxic=bool(prediction),
            toxicity_score=float(toxicity_score),
            confidence=float(confidence),
            model_used=model_name,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchCommentRequest):
    """Predict toxicity for multiple comments."""
    start_time = datetime.now()
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts allowed per batch request")
    
    # Determine which model to use
    model_name = request.model_name
    if model_name is None:
        available_models = list(model_manager.models.keys())
        if not available_models:
            all_models = model_manager.get_available_models()
            if all_models:
                model_name = all_models[0]
                model_manager.load_model(model_name)
            else:
                raise HTTPException(status_code=503, detail="No models available")
        else:
            model_name = available_models[0]
    
    # Load model if not already loaded
    if model_name not in model_manager.models:
        try:
            model_manager.load_model(model_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    model = model_manager.models[model_name]
    
    try:
        predictions = []
        
        # Preprocess all texts
        cleaned_texts = [text_preprocessor.preprocess_text(text) for text in request.texts]
        
        # Extract features for all texts
        features = feature_extractor.extract_tfidf_features(cleaned_texts)
        
        # Make predictions for all texts
        batch_predictions = model.predict(features)
        batch_probabilities = model.predict_proba(features)
        
        for i, text in enumerate(request.texts):
            prediction = batch_predictions[i]
            probabilities = batch_probabilities[i]
            
            # Handle different probability formats
            if len(probabilities.shape) > 0 and len(probabilities) > 1:
                toxicity_score = probabilities[1]
            else:
                toxicity_score = probabilities if probabilities.ndim == 0 else probabilities[0]
            
            confidence = abs(toxicity_score - 0.5) * 2
            
            predictions.append(PredictionResponse(
                text=text,
                is_toxic=bool(prediction),
                toxicity_score=float(toxicity_score),
                confidence=float(confidence),
                model_used=model_name,
                processing_time_ms=0  # Will be set at batch level
            ))
        
        # Update usage stats
        model_manager.model_usage_stats[model_name] += len(request.texts)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get API usage statistics."""
    return {
        "loaded_models": list(model_manager.models.keys()),
        "available_models": model_manager.get_available_models(),
        "model_usage_stats": model_manager.model_usage_stats,
        "total_predictions": sum(model_manager.model_usage_stats.values()),
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model from memory."""
    if model_name in model_manager.models:
        del model_manager.models[model_name]
        return {"message": f"Model {model_name} unloaded successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} is not loaded")

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )