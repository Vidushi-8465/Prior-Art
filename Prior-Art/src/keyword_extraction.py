"""
Keyword Extraction Module
--------------------------
Implements multiple keyword extraction techniques including YAKE, RAKE, and KeyBERT.
"""

import yake
from rake_nltk import Rake
from keybert import KeyBERT
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class KeywordExtractor:
    """
    Extract keywords using various algorithms.
    """
    
    def __init__(self):
        """Initialize all keyword extractors."""
        # YAKE extractor
        self.yake_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # Max n-gram size
            dedupLim=0.9,
            top=20,
            features=None
        )
        
        # RAKE extractor
        self.rake_extractor = Rake()
        
        # KeyBERT extractor (lazy loading to save memory)
        self._keybert_extractor = None
    
    @property
    def keybert_extractor(self):
        """Lazy load KeyBERT model."""
        if self._keybert_extractor is None:
            print("Loading KeyBERT model (this may take a moment)...")
            self._keybert_extractor = KeyBERT()
        return self._keybert_extractor
    
    def extract_with_yake(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using YAKE algorithm.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples (lower score = better)
        """
        keywords = self.yake_extractor.extract_keywords(text)
        return keywords[:top_n]
    
    def extract_with_rake(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using RAKE algorithm.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples (higher score = better)
        """
        self.rake_extractor.extract_keywords_from_text(text)
        keywords_scores = self.rake_extractor.get_ranked_phrases_with_scores()
        return keywords_scores[:top_n]
    
    def extract_with_keybert(self, 
                             text: str, 
                             top_n: int = 10,
                             use_mmr: bool = True) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT (BERT-based semantic extraction).
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            use_mmr: Use Maximal Marginal Relevance for diversity
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            if use_mmr:
                keywords = self.keybert_extractor.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.5,
                    top_n=top_n
                )
            else:
                keywords = self.keybert_extractor.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english',
                    top_n=top_n
                )
            return keywords
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return []
    
    def extract_keywords(self, 
                        text: str, 
                        method: str = "yake",
                        top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Main keyword extraction method.
        
        Args:
            text: Input text
            method: 'yake', 'rake', or 'keybert'
            top_n: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        method = method.lower()
        
        if method == "yake":
            return self.extract_with_yake(text, top_n)
        elif method == "rake":
            return self.extract_with_rake(text, top_n)
        elif method == "keybert":
            return self.extract_with_keybert(text, top_n)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'yake', 'rake', or 'keybert'")
    
    def extract_combined(self, text: str, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract keywords using all methods and return combined results.
        
        Args:
            text: Input text
            top_n: Number of keywords per method
            
        Returns:
            Dictionary with results from each method
        """
        results = {
            'yake': self.extract_with_yake(text, top_n),
            'rake': self.extract_with_rake(text, top_n)
        }
        
        # KeyBERT is optional (slower)
        try:
            results['keybert'] = self.extract_with_keybert(text, top_n)
        except:
            results['keybert'] = []
        
        return results
    
    @staticmethod
    def get_unique_keywords(keyword_lists: Dict[str, List[Tuple[str, float]]], 
                           top_n: int = 15) -> List[str]:
        """
        Combine and deduplicate keywords from multiple methods.
        
        Args:
            keyword_lists: Dictionary of keyword lists from different methods
            top_n: Number of unique keywords to return
            
        Returns:
            List of unique keywords
        """
        all_keywords = []
        for method, keywords in keyword_lists.items():
            all_keywords.extend([kw for kw, score in keywords])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
            
            if len(unique_keywords) >= top_n:
                break
        
        return unique_keywords


if __name__ == "__main__":
    # Example usage
    extractor = KeywordExtractor()
    
    sample_text = """
    A novel machine learning system for automated patent classification using 
    natural language processing and deep neural networks. The invention comprises 
    a transformer-based architecture that analyzes patent documents and extracts 
    technical features. The system uses semantic embeddings to compare prior art 
    and determine novelty scores.
    """
    
    print("Sample Text:")
    print(sample_text)
    print("\n" + "="*70 + "\n")
    
    # YAKE
    print("YAKE Keywords (lower score = more relevant):")
    yake_keywords = extractor.extract_with_yake(sample_text, top_n=8)
    for kw, score in yake_keywords:
        print(f"  {kw:30s} | Score: {score:.4f}")
    
    print("\n" + "="*70 + "\n")
    
    # RAKE
    print("RAKE Keywords (higher score = more relevant):")
    rake_keywords = extractor.extract_with_rake(sample_text, top_n=8)
    for kw, score in rake_keywords:
        print(f"  {kw:30s} | Score: {score:.2f}")
    
    print("\n" + "="*70 + "\n")
    
    # Combined
    print("Combined Unique Keywords:")
    combined = extractor.extract_combined(sample_text, top_n=8)
    unique_kw = extractor.get_unique_keywords(combined, top_n=10)
    for i, kw in enumerate(unique_kw, 1):
        print(f"  {i}. {kw}")