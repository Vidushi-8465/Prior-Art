"""
Text Preprocessing Module
--------------------------
This module handles all text cleaning and preprocessing tasks.
"""

import re
import spacy
from unidecode import unidecode
from typing import List, Optional


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing using spaCy.
    
    Features:
    - Remove special characters and numbers
    - Normalize whitespace
    - Lemmatization
    - Remove stopwords
    - Convert to lowercase
    """
    
    def __init__(self, language_model: str = "en_core_web_sm"):
        """
        Initialize the preprocessor with a spaCy language model.
        
        Args:
            language_model: spaCy model name (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(language_model)
        except OSError:
            print(f"Downloading {language_model}...")
            import os
            os.system(f"python -m spacy download {language_model}")
            self.nlp = spacy.load(language_model)
    
    def clean_text(self, text: str, remove_numbers: bool = True) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Input text to clean
            remove_numbers: Whether to remove numeric characters
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove accented characters
        text = unidecode(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters
        if remove_numbers:
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        else:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Lemmatize text and optionally remove stopwords.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Lemmatized text
        """
        doc = self.nlp(text)
        
        if remove_stopwords:
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and not token.is_punct]
        else:
            tokens = [token.lemma_ for token in doc 
                     if not token.is_punct]
        
        return ' '.join(tokens)
    
    def preprocess(self, 
                   text: str, 
                   clean: bool = True,
                   lemmatize: bool = True,
                   remove_stopwords: bool = True,
                   remove_numbers: bool = False) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            clean: Apply basic cleaning
            lemmatize: Apply lemmatization
            remove_stopwords: Remove stopwords
            remove_numbers: Remove numeric characters
            
        Returns:
            Fully preprocessed text
        """
        if clean:
            text = self.clean_text(text, remove_numbers=remove_numbers)
        
        if lemmatize:
            text = self.lemmatize(text, remove_stopwords=remove_stopwords)
        
        return text
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text (useful for technical concepts).
        
        Args:
            text: Input text
            
        Returns:
            List of noun phrases
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_text = """
    A novel machine learning algorithm for optimizing neural networks 
    using quantum computing principles. The system achieves 99.5% accuracy.
    """
    
    print("Original Text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    print("Cleaned Text:")
    cleaned = preprocessor.clean_text(sample_text)
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    print("Preprocessed (Cleaned + Lemmatized):")
    preprocessed = preprocessor.preprocess(sample_text)
    print(preprocessed)
    print("\n" + "="*50 + "\n")
    
    print("Noun Phrases:")
    phrases = preprocessor.extract_noun_phrases(sample_text)
    print(phrases)