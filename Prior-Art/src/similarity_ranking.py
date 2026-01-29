"""
Similarity and Ranking Module
------------------------------
Computes document similarity and ranks citations using TF-IDF and BERT embeddings.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SimilarityCalculator:
    """
    Calculate similarity between documents using multiple methods.
    """
    
    def __init__(self):
        """Initialize TF-IDF vectorizer and BERT model."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # BERT model (lazy loading)
        self._bert_model = None
    
    @property
    def bert_model(self):
        """Lazy load BERT model."""
        if self._bert_model is None:
            print("Loading Sentence-BERT model (this may take a moment)...")
            self._bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._bert_model
    
    def compute_tfidf_similarity(self, 
                                 query_text: str, 
                                 corpus_texts: List[str]) -> np.ndarray:
        """
        Compute TF-IDF based cosine similarity.
        
        Args:
            query_text: The input/query document
            corpus_texts: List of documents to compare against
            
        Returns:
            Array of similarity scores (0 to 1)
        """
        # Combine query and corpus for vectorization
        all_texts = [query_text] + corpus_texts
        
        # Compute TF-IDF vectors
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Query is the first document
        query_vector = tfidf_matrix[0:1]
        corpus_vectors = tfidf_matrix[1:]
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, corpus_vectors)[0]
        
        return similarities
    
    def compute_bert_similarity(self, 
                                query_text: str, 
                                corpus_texts: List[str]) -> np.ndarray:
        """
        Compute BERT embedding based cosine similarity.
        
        Args:
            query_text: The input/query document
            corpus_texts: List of documents to compare against
            
        Returns:
            Array of similarity scores (-1 to 1, typically 0 to 1)
        """
        # Encode query and corpus
        query_embedding = self.bert_model.encode([query_text])[0]
        corpus_embeddings = self.bert_model.encode(corpus_texts)
        
        # Compute cosine similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            corpus_embeddings
        )[0]
        
        return similarities
    
    def compute_hybrid_similarity(self,
                                  query_text: str,
                                  corpus_texts: List[str],
                                  tfidf_weight: float = 0.3,
                                  bert_weight: float = 0.7) -> np.ndarray:
        """
        Compute hybrid similarity combining TF-IDF and BERT.
        
        Args:
            query_text: The input/query document
            corpus_texts: List of documents to compare against
            tfidf_weight: Weight for TF-IDF scores
            bert_weight: Weight for BERT scores
            
        Returns:
            Array of combined similarity scores
        """
        tfidf_scores = self.compute_tfidf_similarity(query_text, corpus_texts)
        bert_scores = self.compute_bert_similarity(query_text, corpus_texts)
        
        # Normalize BERT scores to 0-1 range if needed
        bert_scores = (bert_scores + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Weighted combination
        hybrid_scores = (tfidf_weight * tfidf_scores) + (bert_weight * bert_scores)
        
        return hybrid_scores


class CitationRanker:
    """
    Rank citations/documents based on similarity and relevance.
    """
    
    def __init__(self):
        """Initialize the similarity calculator."""
        self.similarity_calc = SimilarityCalculator()
    
    def rank_documents(self,
                      query_text: str,
                      documents: List[Dict[str, str]],
                      method: str = "hybrid",
                      top_n: int = 10) -> List[Tuple[int, Dict[str, str], float]]:
        """
        Rank documents by relevance to query.
        
        Args:
            query_text: The input query/invention description
            documents: List of document dictionaries with 'text' and 'metadata'
            method: 'tfidf', 'bert', or 'hybrid'
            top_n: Number of top documents to return
            
        Returns:
            List of (rank, document, score) tuples
        """
        # Extract text from documents
        corpus_texts = [doc['text'] for doc in documents]
        
        # Compute similarities
        if method.lower() == "tfidf":
            similarities = self.similarity_calc.compute_tfidf_similarity(
                query_text, corpus_texts
            )
        elif method.lower() == "bert":
            similarities = self.similarity_calc.compute_bert_similarity(
                query_text, corpus_texts
            )
        else:  # hybrid
            similarities = self.similarity_calc.compute_hybrid_similarity(
                query_text, corpus_texts
            )
        
        # Create ranked list
        ranked_indices = np.argsort(similarities)[::-1]  # Descending order
        
        ranked_results = []
        for rank, idx in enumerate(ranked_indices[:top_n], 1):
            ranked_results.append((
                rank,
                documents[idx],
                float(similarities[idx])
            ))
        
        return ranked_results
    
    def compute_novelty_score(self,
                             query_text: str,
                             prior_art_docs: List[str],
                             method: str = "hybrid") -> Dict[str, float]:
        """
        Compute a novelty score based on similarity to prior art.
        Higher similarity = Lower novelty.
        
        Args:
            query_text: The new invention description
            prior_art_docs: List of prior art document texts
            method: Similarity computation method
            
        Returns:
            Dictionary with novelty metrics
        """
        # Compute similarities
        if method.lower() == "tfidf":
            similarities = self.similarity_calc.compute_tfidf_similarity(
                query_text, prior_art_docs
            )
        elif method.lower() == "bert":
            similarities = self.similarity_calc.compute_bert_similarity(
                query_text, prior_art_docs
            )
        else:  # hybrid
            similarities = self.similarity_calc.compute_hybrid_similarity(
                query_text, prior_art_docs
            )
        
        # Novelty metrics
        max_similarity = float(np.max(similarities))
        avg_similarity = float(np.mean(similarities))
        
        # Novelty score: inverse of similarity
        novelty_score = 1 - max_similarity
        avg_novelty = 1 - avg_similarity
        
        return {
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'novelty_score': novelty_score,  # 0 (not novel) to 1 (very novel)
            'avg_novelty': avg_novelty,
            'num_prior_art': len(prior_art_docs)
        }


if __name__ == "__main__":
    # Example usage
    calculator = SimilarityCalculator()
    ranker = CitationRanker()
    
    # Sample query
    query = """
    A machine learning system for automated patent analysis using 
    natural language processing and neural networks.
    """
    
    # Sample corpus (simulating prior art)
    corpus = [
        "A neural network system for document classification and analysis.",
        "Methods for extracting information from legal documents using AI.",
        "Patent search system using keyword matching algorithms.",
        "Deep learning for image recognition in medical applications.",
        "Natural language understanding for patent document processing."
    ]
    
    print("Query:")
    print(query)
    print("\n" + "="*70 + "\n")
    
    # TF-IDF Similarity
    print("TF-IDF Similarity Scores:")
    tfidf_scores = calculator.compute_tfidf_similarity(query, corpus)
    for i, score in enumerate(tfidf_scores, 1):
        print(f"  Document {i}: {score:.4f}")
    
    print("\n" + "="*70 + "\n")
    
    # Prepare documents for ranking
    documents = [{'text': doc, 'metadata': {'id': i}} for i, doc in enumerate(corpus, 1)]
    
    # Rank documents
    print("Top 3 Most Similar Documents:")
    ranked = ranker.rank_documents(query, documents, method="tfidf", top_n=3)
    for rank, doc, score in ranked:
        print(f"\n  Rank {rank} (Score: {score:.4f}):")
        print(f"    {doc['text'][:80]}...")
    
    print("\n" + "="*70 + "\n")
    
    # Novelty score
    print("Novelty Analysis:")
    novelty = ranker.compute_novelty_score(query, corpus, method="tfidf")
    print(f"  Max Similarity to Prior Art: {novelty['max_similarity']:.4f}")
    print(f"  Average Similarity: {novelty['avg_similarity']:.4f}")
    print(f"  Novelty Score: {novelty['novelty_score']:.4f}")
    print(f"  Interpretation: ", end="")
    if novelty['novelty_score'] > 0.7:
        print("HIGH novelty - significantly different from prior art")
    elif novelty['novelty_score'] > 0.4:
        print("MODERATE novelty - some similarities exist")
    else:
        print("LOW novelty - very similar to existing prior art")