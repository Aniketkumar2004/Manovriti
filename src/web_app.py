"""
Streamlit web application for toxic comment classification.
Provides an interactive interface for testing models and visualizing results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import List, Dict, Any
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_preprocessing import TextPreprocessor, FeatureExtractor, load_sample_data
    from models import (
        LogisticRegressionModel, RandomForestModel, NaiveBayesModel,
        ModelComparison
    )
    LOCAL_MODE = True
except ImportError:
    LOCAL_MODE = False
    st.warning("Local models not available. Using API mode only.")

# Page configuration
st.set_page_config(
    page_title="Toxic Comment Classification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .toxic-prediction {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .non-toxic-prediction {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

class WebApp:
    """Main web application class."""
    
    def __init__(self):
        self.api_url = st.session_state.api_url
        self.local_models = {}
        self.feature_extractor = None
        self.text_preprocessor = None
        
        if LOCAL_MODE:
            self.initialize_local_models()
    
    def initialize_local_models(self):
        """Initialize local models if available."""
        try:
            self.text_preprocessor = TextPreprocessor()
            self.feature_extractor = FeatureExtractor()
            
            # Initialize some basic models for demonstration
            self.local_models = {
                'Logistic Regression': LogisticRegressionModel(),
                'Random Forest': RandomForestModel(n_estimators=50),
                'Naive Bayes': NaiveBayesModel()
            }
            
        except Exception as e:
            st.error(f"Error initializing local models: {e}")
    
    def check_api_health(self) -> bool:
        """Check if the API is available."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_api_models(self) -> List[Dict]:
        """Get available models from API."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def predict_with_api(self, text: str, model_name: str = None) -> Dict:
        """Make prediction using API."""
        try:
            payload = {"text": text}
            if model_name:
                payload["model_name"] = model_name
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def predict_batch_with_api(self, texts: List[str], model_name: str = None) -> Dict:
        """Make batch predictions using API."""
        try:
            payload = {"texts": texts}
            if model_name:
                payload["model_name"] = model_name
            
            response = requests.post(
                f"{self.api_url}/predict/batch",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def predict_with_local_model(self, text: str, model_name: str) -> Dict:
        """Make prediction using local model."""
        if not LOCAL_MODE or model_name not in self.local_models:
            return {"error": "Local model not available"}
        
        try:
            start_time = time.time()
            
            # Preprocess text
            cleaned_text = self.text_preprocessor.preprocess_text(text)
            
            # For demo purposes, train on sample data if not trained
            model = self.local_models[model_name]
            if not model.is_trained:
                sample_data = load_sample_data()
                from data_preprocessing import prepare_data
                X_train, _, y_train, _ = prepare_data(sample_data, test_size=0.1)
                features = self.feature_extractor.extract_tfidf_features(X_train.tolist())
                model.train(features, y_train)
            
            # Extract features and predict
            features = self.feature_extractor.extract_tfidf_features([cleaned_text])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            toxicity_score = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            confidence = abs(toxicity_score - 0.5) * 2
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "text": text,
                "is_toxic": bool(prediction),
                "toxicity_score": float(toxicity_score),
                "confidence": float(confidence),
                "model_used": model_name,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            return {"error": f"Local prediction failed: {str(e)}"}
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🛡️ Toxic Comment Classification</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # API status indicator
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            api_status = self.check_api_health()
            status_color = "🟢" if api_status else "🔴"
            st.metric("API Status", f"{status_color} {'Online' if api_status else 'Offline'}")
        
        with col2:
            mode = "Local + API" if LOCAL_MODE else "API Only"
            st.metric("Mode", mode)
        
        with col3:
            st.metric("Predictions Made", len(st.session_state.prediction_history))
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # API URL configuration
        api_url = st.sidebar.text_input(
            "API URL",
            value=st.session_state.api_url,
            help="URL of the FastAPI server"
        )
        
        if api_url != st.session_state.api_url:
            st.session_state.api_url = api_url
            self.api_url = api_url
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Model selection
        st.sidebar.header("🤖 Model Selection")
        
        # Get available models
        api_models = self.get_api_models() if self.check_api_health() else []
        local_models = list(self.local_models.keys()) if LOCAL_MODE else []
        
        all_models = []
        if api_models:
            all_models.extend([f"API: {model['name']}" for model in api_models])
        if local_models:
            all_models.extend([f"Local: {model}" for model in local_models])
        
        if all_models:
            selected_model = st.sidebar.selectbox(
                "Choose Model",
                options=all_models,
                help="Select a model for predictions"
            )
        else:
            selected_model = None
            st.sidebar.warning("No models available")
        
        st.sidebar.markdown("---")
        
        # Clear history button
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
        
        return selected_model
    
    def render_single_prediction(self, selected_model):
        """Render single comment prediction interface."""
        st.header("🔍 Single Comment Analysis")
        
        # Text input
        comment_text = st.text_area(
            "Enter a comment to analyze:",
            height=100,
            placeholder="Type your comment here...",
            help="Enter any text comment to check for toxicity"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            predict_button = st.button("🔮 Predict", type="primary")
        
        if predict_button and comment_text.strip() and selected_model:
            with st.spinner("Analyzing comment..."):
                # Determine prediction method
                if selected_model.startswith("API:"):
                    model_name = selected_model.replace("API: ", "")
                    result = self.predict_with_api(comment_text, model_name)
                elif selected_model.startswith("Local:"):
                    model_name = selected_model.replace("Local: ", "")
                    result = self.predict_with_local_model(comment_text, model_name)
                else:
                    result = {"error": "Invalid model selection"}
                
                # Display results
                if "error" not in result:
                    self.display_prediction_result(result)
                    
                    # Add to history
                    st.session_state.prediction_history.append(result)
                else:
                    st.error(f"Prediction failed: {result['error']}")
        
        elif predict_button and not comment_text.strip():
            st.warning("Please enter a comment to analyze.")
        elif predict_button and not selected_model:
            st.warning("Please select a model first.")
    
    def render_batch_prediction(self, selected_model):
        """Render batch prediction interface."""
        st.header("📊 Batch Analysis")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Text Area", "File Upload", "Sample Data"],
            horizontal=True
        )
        
        texts_to_analyze = []
        
        if input_method == "Text Area":
            batch_text = st.text_area(
                "Enter multiple comments (one per line):",
                height=200,
                placeholder="Comment 1\nComment 2\nComment 3\n...",
                help="Enter each comment on a separate line"
            )
            if batch_text.strip():
                texts_to_analyze = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload a CSV file with comments",
                type=['csv'],
                help="CSV file should have a column named 'comment_text'"
            )
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'comment_text' in df.columns:
                        texts_to_analyze = df['comment_text'].dropna().tolist()[:100]  # Limit to 100
                        st.success(f"Loaded {len(texts_to_analyze)} comments from file")
                    else:
                        st.error("CSV file must have a 'comment_text' column")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        elif input_method == "Sample Data":
            if LOCAL_MODE:
                sample_df = load_sample_data()
                texts_to_analyze = sample_df['comment_text'].tolist()
                st.info(f"Using {len(texts_to_analyze)} sample comments")
            else:
                st.warning("Sample data not available in API-only mode")
        
        # Batch prediction
        if texts_to_analyze and st.button("🚀 Analyze Batch", type="primary"):
            if len(texts_to_analyze) > 100:
                st.warning("Limiting analysis to first 100 comments")
                texts_to_analyze = texts_to_analyze[:100]
            
            with st.spinner(f"Analyzing {len(texts_to_analyze)} comments..."):
                # Determine prediction method
                if selected_model and selected_model.startswith("API:"):
                    model_name = selected_model.replace("API: ", "")
                    result = self.predict_batch_with_api(texts_to_analyze, model_name)
                else:
                    # For local models, predict one by one
                    result = {"predictions": []}
                    if selected_model and selected_model.startswith("Local:"):
                        model_name = selected_model.replace("Local: ", "")
                        for text in texts_to_analyze:
                            pred = self.predict_with_local_model(text, model_name)
                            if "error" not in pred:
                                result["predictions"].append(pred)
                
                # Display batch results
                if "error" not in result and result.get("predictions"):
                    self.display_batch_results(result["predictions"])
                else:
                    st.error("Batch prediction failed")
    
    def display_prediction_result(self, result: Dict):
        """Display a single prediction result."""
        is_toxic = result.get("is_toxic", False)
        toxicity_score = result.get("toxicity_score", 0.0)
        confidence = result.get("confidence", 0.0)
        
        # Prediction box
        box_class = "toxic-prediction" if is_toxic else "non-toxic-prediction"
        status_emoji = "⚠️" if is_toxic else "✅"
        status_text = "TOXIC" if is_toxic else "NON-TOXIC"
        
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h3>{status_emoji} {status_text}</h3>
            <p><strong>Comment:</strong> "{result.get('text', '')}"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toxicity Score", f"{toxicity_score:.3f}")
        with col2:
            st.metric("Confidence", f"{confidence:.3f}")
        with col3:
            st.metric("Model Used", result.get("model_used", "Unknown"))
        with col4:
            processing_time = result.get("processing_time_ms", 0)
            st.metric("Processing Time", f"{processing_time:.1f}ms")
        
        # Visualization
        self.create_prediction_gauge(toxicity_score)
    
    def display_batch_results(self, predictions: List[Dict]):
        """Display batch prediction results."""
        st.success(f"Analyzed {len(predictions)} comments successfully!")
        
        # Create DataFrame
        df = pd.DataFrame(predictions)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        toxic_count = df['is_toxic'].sum()
        avg_toxicity = df['toxicity_score'].mean()
        avg_confidence = df['confidence'].mean()
        avg_processing_time = df['processing_time_ms'].mean()
        
        with col1:
            st.metric("Toxic Comments", f"{toxic_count}/{len(df)}")
        with col2:
            st.metric("Avg Toxicity Score", f"{avg_toxicity:.3f}")
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        with col4:
            st.metric("Avg Processing Time", f"{avg_processing_time:.1f}ms")
        
        # Visualizations
        self.create_batch_visualizations(df)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_only_toxic = st.checkbox("Show only toxic comments")
        with col2:
            min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.1)
        
        # Filter data
        filtered_df = df.copy()
        if show_only_toxic:
            filtered_df = filtered_df[filtered_df['is_toxic'] == True]
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        
        # Display table
        display_df = filtered_df[['text', 'is_toxic', 'toxicity_score', 'confidence']].copy()
        display_df.columns = ['Comment', 'Is Toxic', 'Toxicity Score', 'Confidence']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="toxic_comment_analysis.csv",
            mime="text/csv"
        )
    
    def create_prediction_gauge(self, toxicity_score: float):
        """Create a gauge chart for toxicity score."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=toxicity_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Toxicity Score"},
            delta={'reference': 0.5},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_batch_visualizations(self, df: pd.DataFrame):
        """Create visualizations for batch results."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Toxicity distribution
            fig = px.histogram(
                df, 
                x='toxicity_score', 
                nbins=20,
                title="Toxicity Score Distribution",
                color_discrete_sequence=['skyblue']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Toxic vs Non-toxic pie chart
            toxic_counts = df['is_toxic'].value_counts()
            fig = px.pie(
                values=toxic_counts.values,
                names=['Non-toxic', 'Toxic'],
                title="Comment Classification",
                color_discrete_sequence=['lightgreen', 'lightcoral']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence vs Toxicity scatter plot
        fig = px.scatter(
            df,
            x='confidence',
            y='toxicity_score',
            color='is_toxic',
            title="Confidence vs Toxicity Score",
            color_discrete_map={True: 'red', False: 'green'},
            hover_data=['text']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_history(self):
        """Render prediction history."""
        st.header("📈 Prediction History")
        
        if not st.session_state.prediction_history:
            st.info("No predictions made yet. Try analyzing some comments!")
            return
        
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_predictions = len(history_df)
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            toxic_percentage = (history_df['is_toxic'].sum() / len(history_df)) * 100
            st.metric("Toxic Percentage", f"{toxic_percentage:.1f}%")
        
        with col3:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        # Timeline chart
        if len(history_df) > 1:
            history_df['prediction_id'] = range(1, len(history_df) + 1)
            
            fig = px.line(
                history_df,
                x='prediction_id',
                y='toxicity_score',
                color='is_toxic',
                title="Toxicity Scores Over Time",
                color_discrete_map={True: 'red', False: 'green'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent predictions table
        st.subheader("Recent Predictions")
        recent_df = history_df.tail(10)[['text', 'is_toxic', 'toxicity_score', 'confidence', 'model_used']]
        recent_df.columns = ['Comment', 'Is Toxic', 'Toxicity Score', 'Confidence', 'Model']
        
        st.dataframe(recent_df, use_container_width=True)
    
    def run(self):
        """Run the main application."""
        self.render_header()
        selected_model = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 History"])
        
        with tab1:
            self.render_single_prediction(selected_model)
        
        with tab2:
            self.render_batch_prediction(selected_model)
        
        with tab3:
            self.render_history()

def main():
    """Main function to run the Streamlit app."""
    app = WebApp()
    app.run()

if __name__ == "__main__":
    main()