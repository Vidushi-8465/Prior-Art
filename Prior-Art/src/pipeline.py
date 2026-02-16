"""
Enhanced Patent Analysis Pipeline
----------------------------------
Takes PDF input, extracts metadata, keywords with expansions, and outputs CSV.
"""

import os
from typing import List, Dict
from datetime import datetime

from patent_pdf_extractor import PatentPDFExtractor
from preprocessing import TextPreprocessor
from summarization import ModernTextSummarizer
from keyword_extraction import KeywordExtractor
from keyword_expander import KeywordExpander
from csv_output_generator import CSVOutputGenerator
# Similarity ranking 

class PatentAnalysisPipeline:
    """
    Complete pipeline for patent PDF analysis with CSV output.
    
    Workflow:
    1. Extract text and metadata from PDF
    2. Preprocess text
    3. Generate summary
    4. Extract keywords
    5. Expand keywords with synonyms/full forms
    6. Generate CSV output
    """
    
    def __init__(self, output_dir: str = "../data/output"):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing Patent Analysis Pipeline...")
        self.pdf_extractor = PatentPDFExtractor()
        self.preprocessor = TextPreprocessor()
        self.summarizer = ModernTextSummarizer()
        self.keyword_extractor = KeywordExtractor()
        self.keyword_expander = KeywordExpander()
        self.csv_generator = CSVOutputGenerator()
        print("Pipeline ready!\n")
    
    def analyze_single_pdf(self, pdf_path: str, num_keywords: int = 15) -> Dict:
        """
        Analyze a single patent PDF.
        
        Args:
            pdf_path: Path to PDF file
            num_keywords: Number of keywords to extract
            
        Returns:
            Dictionary with all analysis results
        """
        print(f"\nAnalyzing: {os.path.basename(pdf_path)}")
        print("-" * 70)
        
        # Step 1: Extract PDF content and metadata
        print("Step 1: Extracting PDF content...")
        pdf_data = self.pdf_extractor.extract_all(pdf_path)
        
        text = pdf_data['text']
        if not text or len(text) < 100:
            print("  Warning: Extracted text is too short!")
            return None
        
        print(f"  ✓ Extracted {len(text)} characters from {pdf_data['num_pages']} pages")
        print(f"  ✓ Title: {pdf_data['title'][:80] if pdf_data['title'] else 'Not found'}...")
        print(f"  ✓ Year: {pdf_data['publication_year'] or 'Not found'}")
        print(f"  ✓ IPC Codes: {len(pdf_data['ipc_codes'])} found")
        print(f"  ✓ CPC Codes: {len(pdf_data['cpc_codes'])} found")
        print(f"  ✓ Citations: {len(pdf_data['citations'])} found")
        
        # Step 2: Generate summary
        print("\nStep 2: Generating summary...")
        if pdf_data['abstract']:
            summary = pdf_data['abstract']
            print("  ✓ Using extracted abstract as summary")
        else:
            # Use first ~1000 chars for summary if no abstract
            summary_text = text[:1000] if len(text) > 1000 else text
            summary = self.summarizer.summarize_with_textrank(summary_text, sentence_count=3)
            print("  ✓ Generated summary from text")
        
        # Step 3: Extract keywords
        print("\nStep 3: Extracting keywords...")
        try:
            # Use combined approach for best results
            keywords_dict = self.keyword_extractor.extract_combined(text, top_n=num_keywords)
            keywords_list = self.keyword_extractor.get_unique_keywords(keywords_dict, top_n=num_keywords)
            print(f"  ✓ Extracted {len(keywords_list)} keywords")
        except Exception as e:
            print(f"  Warning: Keyword extraction error: {e}")
            # Fallback to YAKE only
            try:
                keywords_yake = self.keyword_extractor.extract_with_yake(text, top_n=num_keywords)
                keywords_list = [kw for kw, score in keywords_yake]
                print(f"  ✓ Extracted {len(keywords_list)} keywords (YAKE only)")
            except:
                keywords_list = []
                print("  ✗ Could not extract keywords")
        
        # Step 4: Expand keywords with synonyms/full forms
        print("\nStep 4: Expanding keywords...")
        expanded_keywords = self.keyword_expander.expand_keywords(keywords_list, text)
        keywords_formatted = self.keyword_expander.format_expanded_keywords(expanded_keywords)
        print(f"  ✓ Expanded {len(expanded_keywords)} keywords")
        
        # Compile results
        result = {
            'filename': pdf_data['filename'],
            'filepath': pdf_path,
            'title': pdf_data['title'] or '',
            'abstract': pdf_data['abstract'] or summary,
            'summary': summary,
            'publication_year': pdf_data['publication_year'] or '',
            'ipc_codes': pdf_data['ipc_codes'],
            'cpc_codes': pdf_data['cpc_codes'],
            'keywords': keywords_list,
            'keywords_expanded': expanded_keywords,
            'keywords_formatted': keywords_formatted,
            'citations': pdf_data['citations'],
            'num_pages': pdf_data['num_pages'],
            'text_length': len(text)
        }
        
        print("\n✓ Analysis complete!")
        
        return result
    
    def analyze_multiple_pdfs(self, pdf_paths: List[str], num_keywords: int = 15) -> List[Dict]:
        """
        Analyze multiple patent PDFs.
        
        Args:
            pdf_paths: List of paths to PDF files
            num_keywords: Number of keywords to extract per document
            
        Returns:
            List of analysis results
        """
        results = []
        
        print("="*70)
        print(f"PATENT ANALYSIS PIPELINE - {len(pdf_paths)} DOCUMENTS")
        print("="*70)
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"\n[{i}/{len(pdf_paths)}] Processing...")
            
            try:
                result = self.analyze_single_pdf(pdf_path, num_keywords)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        return results
    
    def generate_output_csv(self, results: List[Dict], output_filename: str = None) -> str:
        """
        Generate CSV output from results.
        
        Args:
            results: List of analysis results
            output_filename: Custom output filename (optional)
            
        Returns:
            Path to generated CSV file
        """
        if not results:
            print("No results to save!")
            return None
        
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"patent_analysis_{timestamp}.csv"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Generate CSV
        csv_path = self.csv_generator.generate_csv(results, output_path)
        
        # Print summary
        self.csv_generator.print_summary(csv_path, results)
        
        return csv_path
    
    def run_pipeline(self, 
                     pdf_paths: List[str], 
                     num_keywords: int = 15,
                     output_filename: str = None) -> str:
        """
        Run the complete pipeline: PDF input → Analysis → CSV output.
        
        Args:
            pdf_paths: List of PDF file paths (can be single file or multiple)
            num_keywords: Number of keywords to extract
            output_filename: Custom output filename
            
        Returns:
            Path to generated CSV file
        """
        # Handle single PDF input
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        
        # Analyze all PDFs
        results = self.analyze_multiple_pdfs(pdf_paths, num_keywords)
        
        # Generate CSV output
        csv_path = self.generate_output_csv(results, output_filename)
        
        return csv_path
    
    def print_result_preview(self, result: Dict):
        """
        Print a formatted preview of analysis results.
        
        Args:
            result: Analysis result dictionary
        """
        print("\n" + "="*70)
        print("ANALYSIS PREVIEW")
        print("="*70)
        
        print(f"\nFile: {result['filename']}")
        print(f"Title: {result['title']}")
        print(f"Year: {result['publication_year']}")
        
        print(f"\nAbstract:")
        print(f"  {result['abstract'][:200]}...")
        
        print(f"\nIPC Codes ({len(result['ipc_codes'])}):")
        print(f"  {', '.join(result['ipc_codes'][:5])}")
        
        print(f"\nCPC Codes ({len(result['cpc_codes'])}):")
        print(f"  {', '.join(result['cpc_codes'][:5])}")
        
        print(f"\nKeywords ({len(result['keywords'])}):")
        for i, (kw, expansions) in enumerate(result['keywords_expanded'][:10], 1):
            if len(expansions) > 1:
                print(f"  {i}. {kw} → {', '.join(expansions[1:])}")
            else:
                print(f"  {i}. {kw}")
        
        print(f"\nCitations ({len(result['citations'])}):")
        print(f"  {', '.join(result['citations'][:10])}")
        
        print("="*70)


if __name__ == "__main__":
    # Example usage
    pipeline = PatentAnalysisPipeline(output_dir="../data/output")
    
    # Example: Analyze a single PDF
    # csv_path = pipeline.run_pipeline("path/to/patent.pdf")
    
    # Example: Analyze multiple PDFs
    # pdf_files = ["patent1.pdf", "patent2.pdf", "patent3.pdf"]
    # csv_path = pipeline.run_pipeline(pdf_files, num_keywords=15)
    
    print("Patent Analysis Pipeline ready!")
    print("\nUsage:")
    print("  pipeline.run_pipeline('path/to/patent.pdf')")
    print("  pipeline.run_pipeline(['file1.pdf', 'file2.pdf'])")