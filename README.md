# 🛡️ Toxic Comment Classification Project

A comprehensive machine learning project for detecting toxic comments using both traditional ML and deep learning approaches. This project provides a complete pipeline from data preprocessing to model deployment with web interface and REST API.

## 🌟 Features

- **Multiple ML Models**: Logistic Regression, Random Forest, SVM, Naive Bayes, Gradient Boosting, LSTM, BERT
- **Ensemble Methods**: Combine multiple models for better performance
- **REST API**: FastAPI-based API for model inference
- **Web Interface**: Interactive Streamlit app for testing models
- **Comprehensive Evaluation**: Detailed model comparison and visualization
- **Production Ready**: Docker support, logging, error handling

## 📁 Project Structure

```
toxic-comment-classification/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Text preprocessing and feature extraction
│   ├── models.py               # ML models implementation
│   ├── train_evaluate.py       # Training and evaluation pipeline
│   ├── api.py                  # FastAPI REST API
│   └── web_app.py             # Streamlit web interface
├── examples/
│   ├── basic_usage.py          # Basic usage examples
│   ├── api_client.py           # API client examples
│   └── model_comparison.py     # Model comparison example
├── notebooks/
│   └── exploration.ipynb       # Jupyter notebook for exploration
├── tests/
│   ├── test_preprocessing.py   # Unit tests
│   ├── test_models.py
│   └── test_api.py
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd toxic-comment-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from src.data_preprocessing import load_sample_data, prepare_data, FeatureExtractor
from src.models import LogisticRegressionModel, ModelComparison

# Load sample data
df = load_sample_data()
X_train, X_test, y_train, y_test = prepare_data(df)

# Extract features
feature_extractor = FeatureExtractor()
X_train_features = feature_extractor.extract_tfidf_features(X_train.tolist())
X_test_features = feature_extractor.extract_tfidf_features(X_test.tolist())

# Train model
model = LogisticRegressionModel()
model.train(X_train_features, y_train)

# Make predictions
predictions = model.predict(X_test_features)
probabilities = model.predict_proba(X_test_features)

print(f"Accuracy: {(predictions == y_test).mean():.3f}")
```

### 3. Train Multiple Models

```bash
# Train and evaluate all models
python src/train_evaluate.py --output results

# Train with custom data
python src/train_evaluate.py --data your_data.csv --output custom_results
```

### 4. Start the API Server

```bash
# Start the FastAPI server
python src/api.py

# Or using uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Launch Web Interface

```bash
# Start the Streamlit app
streamlit run src/web_app.py
```

## 📊 Model Performance

The project includes several models with different strengths:

| Model | Accuracy | Precision | Recall | F1-Score | Speed |
|-------|----------|-----------|--------|----------|-------|
| Logistic Regression | 0.85 | 0.83 | 0.87 | 0.85 | Fast |
| Random Forest | 0.87 | 0.85 | 0.89 | 0.87 | Medium |
| SVM | 0.86 | 0.84 | 0.88 | 0.86 | Slow |
| Naive Bayes | 0.82 | 0.80 | 0.85 | 0.82 | Fast |
| Gradient Boosting | 0.88 | 0.86 | 0.90 | 0.88 | Medium |
| Ensemble | 0.89 | 0.87 | 0.91 | 0.89 | Medium |

*Note: Performance metrics are examples and will vary based on your data.*

## 🔧 API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test comment"}'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Comment 1", "Comment 2", "Comment 3"]}'
```

### Python API Client

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This is a test comment"}
)
result = response.json()
print(f"Is toxic: {result['is_toxic']}")
print(f"Toxicity score: {result['toxicity_score']:.3f}")
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t toxic-comment-classifier .

# Run the container
docker run -p 8000:8000 toxic-comment-classifier
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📈 Web Interface Features

The Streamlit web interface provides:

- **Single Comment Analysis**: Test individual comments
- **Batch Processing**: Upload CSV files or paste multiple comments
- **Model Comparison**: Compare different models side-by-side
- **Visualization**: Interactive charts and graphs
- **History Tracking**: Keep track of all predictions
- **Export Results**: Download results as CSV

Access the web interface at `http://localhost:8501` after running:

```bash
streamlit run src/web_app.py
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📝 Configuration

### Model Configuration

Create a `model_config.json` file to customize model parameters:

```json
{
  "logistic_regression": {
    "C": 1.0,
    "max_iter": 1000
  },
  "random_forest": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "svm": {
    "C": 1.0,
    "kernel": "rbf"
  }
}
```

Use it with the training script:

```bash
python src/train_evaluate.py --config model_config.json
```

## 🔍 Data Format

The project expects data in CSV format with the following columns:

- `comment_text`: The text content to classify
- `toxic`: Binary label (0 for non-toxic, 1 for toxic)

Example:

```csv
comment_text,toxic
"This is a great post!",0
"You are an idiot",1
"Thanks for sharing",0
```

## 🎯 Use Cases

This project is suitable for:

- **Content Moderation**: Automatically flag potentially toxic comments
- **Social Media Monitoring**: Monitor discussions for harmful content
- **Community Management**: Help moderators prioritize review queues
- **Research**: Study patterns in online toxicity
- **Education**: Learn about NLP and text classification

## 🛠️ Customization

### Adding New Models

1. Create a new model class inheriting from `BaseModel`:

```python
from src.models import BaseModel

class YourCustomModel(BaseModel):
    def __init__(self):
        super().__init__("Your Custom Model")
        # Initialize your model here
    
    def train(self, X_train, y_train, **kwargs):
        # Implement training logic
        pass
    
    def predict(self, X):
        # Implement prediction logic
        pass
    
    def predict_proba(self, X):
        # Implement probability prediction
        pass
```

2. Add it to the training pipeline in `train_evaluate.py`

### Custom Preprocessing

Modify the `TextPreprocessor` class in `data_preprocessing.py` to add custom preprocessing steps:

```python
def custom_preprocessing_step(self, text: str) -> str:
    # Your custom preprocessing logic
    return processed_text
```

## 📊 Monitoring and Logging

The project includes comprehensive logging:

- **API Logs**: Request/response logging with timing
- **Model Performance**: Training metrics and evaluation results
- **Error Tracking**: Detailed error messages and stack traces

Logs are written to both console and files in the `logs/` directory.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [PyTorch](https://pytorch.org/), and [Transformers](https://huggingface.co/transformers/)
- Web interface powered by [Streamlit](https://streamlit.io/)
- API built with [FastAPI](https://fastapi.tiangolo.com/)
- Visualizations created with [Plotly](https://plotly.com/python/)

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🔄 Version History

- **v1.0.0**: Initial release with basic models and API
- **v1.1.0**: Added web interface and batch processing
- **v1.2.0**: Added ensemble methods and BERT support
- **v1.3.0**: Added Docker support and comprehensive testing

---

**Happy Classifying! 🛡️**


