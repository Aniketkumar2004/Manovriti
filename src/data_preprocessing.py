"""
Data preprocessing module for toxic comment classification.
Handles text cleaning, tokenization, and feature extraction.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Comprehensive text preprocessing for toxic comment classification."""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation but keep important ones for context
        text = re.sub(r'[^\w\s!?.]', '', text)
        
        # Remove repeated characters (e.g., 'sooooo' -> 'so')
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        return text.strip()
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """Tokenize text and apply preprocessing steps."""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove single characters and numbers
        tokens = [token for token in tokens if len(token) > 1 and not token.isdigit()]
        
        # Lemmatize if specified
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """Complete preprocessing pipeline."""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_process(cleaned_text)
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocess text in a DataFrame."""
        logger.info(f"Preprocessing {len(df)} comments...")
        
        df_processed = df.copy()
        df_processed[f'{text_column}_cleaned'] = df_processed[text_column].apply(
            self.preprocess_text
        )
        
        # Add text statistics
        df_processed[f'{text_column}_length'] = df_processed[text_column].str.len()
        df_processed[f'{text_column}_word_count'] = df_processed[text_column].str.split().str.len()
        df_processed[f'{text_column}_cleaned_length'] = df_processed[f'{text_column}_cleaned'].str.len()
        df_processed[f'{text_column}_cleaned_word_count'] = df_processed[f'{text_column}_cleaned'].str.split().str.len()
        
        # Add toxicity indicators
        df_processed[f'{text_column}_has_caps'] = df_processed[text_column].str.contains(r'[A-Z]{3,}', na=False)
        df_processed[f'{text_column}_exclamation_count'] = df_processed[text_column].str.count('!')
        df_processed[f'{text_column}_question_count'] = df_processed[text_column].str.count(r'\?')
        
        logger.info("Preprocessing completed!")
        return df_processed


class FeatureExtractor:
    """Extract features for machine learning models."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
    def extract_tfidf_features(self, texts: List[str], max_features: int = 10000,
                              ngram_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
        """Extract TF-IDF features."""
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            features = self.tfidf_vectorizer.transform(texts)
        
        return features.toarray()
    
    def extract_count_features(self, texts: List[str], max_features: int = 10000,
                              ngram_range: Tuple[int, int] = (1, 2)) -> np.ndarray:
        """Extract count-based features."""
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            features = self.count_vectorizer.fit_transform(texts)
        else:
            features = self.count_vectorizer.transform(texts)
        
        return features.toarray()
    
    def get_feature_names(self, vectorizer_type: str = 'tfidf') -> List[str]:
        """Get feature names from the vectorizer."""
        if vectorizer_type == 'tfidf' and self.tfidf_vectorizer:
            return self.tfidf_vectorizer.get_feature_names_out().tolist()
        elif vectorizer_type == 'count' and self.count_vectorizer:
            return self.count_vectorizer.get_feature_names_out().tolist()
        else:
            return []


def load_sample_data() -> pd.DataFrame:
    """Load or create sample toxic comment data for testing."""
    # Sample data for demonstration
    sample_data = {
        'comment_text': [
            "You are such an idiot!",
            "This is a great post, thanks for sharing!",
            "I hate you so much, you stupid person",
            "Nice work on this project!",
            "Go kill yourself, nobody likes you",
            "I disagree with your opinion, but I respect it",
            "You're the worst person ever, I hope bad things happen to you",
            "Thanks for the helpful information!",
            "This is complete garbage and you should be ashamed",
            "I found this very informative and well-written",
            "You moron, how can you be so dumb?",
            "Great explanation, very clear and concise!",
            "I wish you would just disappear forever",
            "This helped me understand the concept better",
            "You're pathetic and worthless",
            "Excellent work, keep it up!",
            "I hope you get what you deserve, loser",
            "Thank you for taking the time to explain this",
            "You make me sick with your stupidity",
            "This is exactly what I was looking for!"
        ],
        'toxic': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    return pd.DataFrame(sample_data)


def prepare_data(df: pd.DataFrame, text_column: str = 'comment_text', 
                target_column: str = 'toxic', test_size: float = 0.2,
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for training and testing."""
    # Preprocess text
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df, text_column)
    
    # Split data
    X = df_processed[f'{text_column}_cleaned']
    y = df_processed[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Toxic comments in training: {y_train.sum()}/{len(y_train)} ({y_train.mean():.2%})")
    logger.info(f"Toxic comments in test: {y_test.sum()}/{len(y_test)} ({y_test.mean():.2%})")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    df = load_sample_data()
    print("Sample data loaded:")
    print(df.head())
    
    # Preprocess data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Extract features
    feature_extractor = FeatureExtractor()
    X_train_tfidf = feature_extractor.extract_tfidf_features(X_train.tolist())
    X_test_tfidf = feature_extractor.extract_tfidf_features(X_test.tolist())
    
    print(f"\nTF-IDF features shape: {X_train_tfidf.shape}")
    print("Preprocessing completed successfully!")