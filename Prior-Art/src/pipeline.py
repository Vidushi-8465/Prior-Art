"""
Prior Art Search Pipeline
--------------------------
Main pipeline that integrates all components for end-to-end prior art analysis.
"""

import os
import json
from typing import List, Dict, Optional, Union
from datetime import datetime

from pdf_extractor import PDFExtractor
from preprocessing import TextPreprocessor
from summarization import TextSummarizer, ModernTextSummarizer
from keyword_extraction import KeywordExtractor
from similarity_ranking import CitationRanker


class PriorArtPipeline:
    """
    Complete pipeline for prior art search and analysis.
    
    Workflow:
    1. Extract text from PDF or accept direct input
    2. Preprocess and clean the text
    3. Summarize the invention description
    4. Extract keywords
    5. Compare against prior art corpus
    6. Rank results by similarity
    7. Generate novelty report
    """
    
    def __init__(self, output_dir: str = "../data/output"):
        """
        Initialize the pipeline with all components.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize all components
        print("Initializing pipeline components...")
        self.pdf_extractor = PDFExtractor()
        self.preprocessor = TextPreprocessor()
        self.summarizer = ModernTextSummarizer()
        self.keyword_extractor = KeywordExtractor()
        self.citation_ranker = CitationRanker()
        print("Pipeline ready!\n")
    
    def process_input(self, 
                      input_data: Union[str, Dict],
                      is_file: bool = False) -> Dict:
        """
        Process input text or PDF file.
        
        Args:
            input_data: Either text string or file path
            is_file: True if input_data is a file path
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if is_file:
            # Extract from PDF
            result = self.pdf_extractor.extract(input_data, method="pdfplumber")
        else:
            # Direct text input
            result = self.pdf_extractor.extract_from_text_input(input_data)
        
        return result
    
    def analyze_invention(self,
                         invention_text: str,
                         summarize: bool = True,
                         extract_keywords: bool = True) -> Dict:
        """
        Analyze an invention description.
        
        Args:
            invention_text: The invention description text
            summarize: Whether to generate a summary
            extract_keywords: Whether to extract keywords
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'original_text': invention_text,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Preprocess
        print("Step 1: Preprocessing text...")
        cleaned_text = self.preprocessor.clean_text(invention_text)
        preprocessed_text = self.preprocessor.preprocess(
            invention_text, 
            lemmatize=False,  # Keep original for summary
            remove_stopwords=False
        )
        results['cleaned_text'] = cleaned_text
        
        # Step 2: Summarize
        if summarize:
            print("Step 2: Generating summary...")
            summary = self.summarizer.summarize_with_textrank(
                invention_text, 
                sentence_count=3
            )
            results['summary'] = summary
        
        # Step 3: Extract keywords
        if extract_keywords:
            print("Step 3: Extracting keywords...")
            keywords_combined = self.keyword_extractor.extract_combined(
                invention_text, 
                top_n=10
            )
            
            # Get unique keywords
            unique_keywords = self.keyword_extractor.get_unique_keywords(
                keywords_combined, 
                top_n=15
            )
            
            results['keywords'] = {
                'yake': keywords_combined.get('yake', []),
                'rake': keywords_combined.get('rake', []),
                'keybert': keywords_combined.get('keybert', []),
                'unique_keywords': unique_keywords
            }
        
        return results
    
    def compare_with_prior_art(self,
                               invention_text: str,
                               prior_art_documents: List[Dict[str, str]],
                               method: str = "hybrid",
                               top_n: int = 10) -> Dict:
        """
        Compare invention with prior art and rank results.
        
        Args:
            invention_text: The invention description
            prior_art_documents: List of prior art documents with 'text' and 'metadata'
            method: Similarity method ('tfidf', 'bert', or 'hybrid')
            top_n: Number of top results to return
            
        Returns:
            Dictionary with ranking results and novelty score
        """
        print(f"\nStep 4: Comparing with {len(prior_art_documents)} prior art documents...")
        
        # Rank documents
        ranked_results = self.citation_ranker.rank_documents(
            invention_text,
            prior_art_documents,
            method=method,
            top_n=top_n
        )
        
        # Compute novelty score
        prior_art_texts = [doc['text'] for doc in prior_art_documents]
        novelty_metrics = self.citation_ranker.compute_novelty_score(
            invention_text,
            prior_art_texts,
            method=method
        )
        
        return {
            'ranked_citations': ranked_results,
            'novelty_metrics': novelty_metrics,
            'similarity_method': method,
            'total_prior_art': len(prior_art_documents)
        }
    
    def run_full_pipeline(self,
                         invention_input: Union[str, Dict],
                         prior_art_docs: List[Dict[str, str]],
                         is_file: bool = False,
                         similarity_method: str = "hybrid",
                         save_results: bool = True) -> Dict:
        """
        Run the complete prior art search pipeline.
        
        Args:
            invention_input: Invention text or PDF path
            prior_art_docs: List of prior art documents
            is_file: True if invention_input is a file path
            similarity_method: Method for similarity computation
            save_results: Whether to save results to file
            
        Returns:
            Complete analysis results
        """
        print("="*70)
        print("PRIOR ART SEARCH PIPELINE")
        print("="*70 + "\n")
        
        # Process input
        print("Processing input...")
        input_data = self.process_input(invention_input, is_file=is_file)
        invention_text = input_data['text']
        
        # Analyze invention
        analysis_results = self.analyze_invention(
            invention_text,
            summarize=True,
            extract_keywords=True
        )
        
        # Compare with prior art
        comparison_results = self.compare_with_prior_art(
            invention_text,
            prior_art_docs,
            method=similarity_method,
            top_n=10
        )
        
        # Combine all results
        final_results = {
            'input_metadata': {
                'filename': input_data.get('filename', 'direct_input'),
                'num_pages': input_data.get('num_pages', 1),
                'input_length': len(invention_text)
            },
            'analysis': analysis_results,
            'prior_art_comparison': comparison_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        if save_results:
            output_file = os.path.join(
                self.output_dir, 
                f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        
        return final_results
    
    def print_results(self, results: Dict):
        """
        Pretty print the analysis results.
        
        Args:
            results: Results dictionary from run_full_pipeline
        """
        print("\n" + "="*70)
        print("ANALYSIS RESULTS")
        print("="*70 + "\n")
        
        # Summary
        if 'summary' in results['analysis']:
            print("SUMMARY:")
            print(results['analysis']['summary'])
            print("\n" + "-"*70 + "\n")
        
        # Keywords
        if 'keywords' in results['analysis']:
            print("TOP KEYWORDS:")
            for i, kw in enumerate(results['analysis']['keywords']['unique_keywords'], 1):
                print(f"  {i:2d}. {kw}")
            print("\n" + "-"*70 + "\n")
        
        # Novelty Score
        novelty = results['prior_art_comparison']['novelty_metrics']
        print("NOVELTY ASSESSMENT:")
        print(f"  Novelty Score: {novelty['novelty_score']:.2%}")
        print(f"  Max Similarity to Prior Art: {novelty['max_similarity']:.2%}")
        print(f"  Average Similarity: {novelty['avg_similarity']:.2%}")
        
        if novelty['novelty_score'] > 0.7:
            assessment = "HIGH - Significantly different from prior art"
        elif novelty['novelty_score'] > 0.4:
            assessment = "MODERATE - Some similarities exist"
        else:
            assessment = "LOW - Very similar to existing prior art"
        print(f"  Assessment: {assessment}")
        print("\n" + "-"*70 + "\n")
        
        # Top Citations
        print("TOP 5 MOST SIMILAR PRIOR ART:")
        ranked = results['prior_art_comparison']['ranked_citations'][:5]
        for rank, doc, score in ranked:
            print(f"\n  Rank {rank} | Similarity: {score:.2%}")
            print(f"  {doc['text'][:150]}...")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    # Example usage
    pipeline = PriorArtPipeline(output_dir="../data/output")
    
    # Sample invention description
    invention_description = """
    A novel system and method for real-time emotion detection in video calls using 
    deep learning. The system employs a convolutional neural network to analyze 
    facial expressions and voice patterns simultaneously. The invention includes a 
    multimodal fusion architecture that combines visual and audio features to 
    achieve 95% accuracy in emotion classification. The system can detect seven 
    basic emotions: happiness, sadness, anger, surprise, fear, disgust, and neutral. 
    Applications include mental health monitoring, customer service quality 
    assessment, and virtual therapy sessions.
    """
    
    # Sample prior art (in real scenario, this would be from a database)
    prior_art = [
        {
            'text': 'A facial recognition system using convolutional neural networks for security applications.',
            'metadata': {'id': 'P001', 'year': 2020}
        },
        {
            'text': 'Methods for emotion detection from speech signals using machine learning algorithms.',
            'metadata': {'id': 'P002', 'year': 2019}
        },
        {
            'text': 'Real-time video analysis system for detecting human emotions using deep learning.',
            'metadata': {'id': 'P003', 'year': 2021}
        },
        {
            'text': 'Multimodal sentiment analysis combining text, audio and video features.',
            'metadata': {'id': 'P004', 'year': 2022}
        },
        {
            'text': 'Image classification system using neural networks for medical diagnosis.',
            'metadata': {'id': 'P005', 'year': 2018}
        }
    ]
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        invention_input=invention_description,
        prior_art_docs=prior_art,
        is_file=False,
        similarity_method="tfidf",  # Use 'hybrid' for best results
        save_results=True
    )
    
    # Print results
    pipeline.print_results(results)