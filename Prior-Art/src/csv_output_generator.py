"""
CSV Output Generator
--------------------
Generates CSV output with patent information in the required format.
"""

import csv
import os
from typing import List, Dict
from datetime import datetime


class CSVOutputGenerator:
    """
    Generate CSV output in the specified format:
    title | abstract | ipc_codes | cpc_codes | keywords | citations | publication_year
    """
    
    @staticmethod
    def generate_csv(results: List[Dict], output_path: str = None) -> str:
        """
        Generate CSV file from analysis results.
        
        Args:
            results: List of dictionaries containing patent analysis results
            output_path: Path to save CSV file (optional)
            
        Returns:
            Path to generated CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"patent_analysis_{timestamp}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # CSV headers
        headers = [
            'title',
            'abstract',
            'ipc_codes',
            'cpc_codes',
            'keywords',
            'citations',
            'publication_year'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for result in results:
                # Format lists as comma-separated strings
                row = {
                    'title': result.get('title', ''),
                    'abstract': result.get('abstract', ''),
                    'ipc_codes': ', '.join(result.get('ipc_codes', [])) if result.get('ipc_codes') else '',
                    'cpc_codes': ', '.join(result.get('cpc_codes', [])) if result.get('cpc_codes') else '',
                    'keywords': result.get('keywords_formatted', ''),
                    'citations': ', '.join(result.get('citations', [])) if result.get('citations') else '',
                    'publication_year': result.get('publication_year', '')
                }
                
                writer.writerow(row)
        
        return output_path
    
    @staticmethod
    def generate_detailed_csv(results: List[Dict], output_path: str = None) -> str:
        """
        Generate a more detailed CSV with additional columns.
        
        Args:
            results: List of dictionaries containing patent analysis results
            output_path: Path to save CSV file (optional)
            
        Returns:
            Path to generated CSV file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"patent_analysis_detailed_{timestamp}.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Extended headers
        headers = [
            'filename',
            'title',
            'abstract',
            'ipc_codes',
            'cpc_codes',
            'keywords',
            'keywords_expanded',
            'citations',
            'publication_year',
            'num_pages',
            'summary'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for result in results:
                row = {
                    'filename': result.get('filename', ''),
                    'title': result.get('title', ''),
                    'abstract': result.get('abstract', ''),
                    'ipc_codes': ', '.join(result.get('ipc_codes', [])) if result.get('ipc_codes') else '',
                    'cpc_codes': ', '.join(result.get('cpc_codes', [])) if result.get('cpc_codes') else '',
                    'keywords': ', '.join(result.get('keywords', [])) if result.get('keywords') else '',
                    'keywords_expanded': result.get('keywords_formatted', ''),
                    'citations': ', '.join(result.get('citations', [])) if result.get('citations') else '',
                    'publication_year': result.get('publication_year', ''),
                    'num_pages': result.get('num_pages', ''),
                    'summary': result.get('summary', '')
                }
                
                writer.writerow(row)
        
        return output_path
    
    @staticmethod
    def print_summary(csv_path: str, results: List[Dict]):
        """
        Print a summary of the generated CSV.
        
        Args:
            csv_path: Path to the generated CSV
            results: List of results that were written
        """
        print("\n" + "="*70)
        print("CSV GENERATION COMPLETE")
        print("="*70)
        print(f"\nFile saved: {csv_path}")
        print(f"Total records: {len(results)}")
        print("\nSample data:")
        print("-"*70)
        
        if results:
            first_result = results[0]
            print(f"Title: {first_result.get('title', 'N/A')[:80]}...")
            print(f"Year: {first_result.get('publication_year', 'N/A')}")
            print(f"IPC Codes: {', '.join(first_result.get('ipc_codes', [])[:3])}")
            print(f"CPC Codes: {', '.join(first_result.get('cpc_codes', [])[:3])}")
            print(f"Keywords: {len(first_result.get('keywords', []))} found")
            print(f"Citations: {len(first_result.get('citations', []))} found")
        
        print("="*70)


if __name__ == "__main__":
    # Test with sample data
    generator = CSVOutputGenerator()
    
    sample_results = [
        {
            'title': 'System for automated text analysis',
            'abstract': 'A novel system for analyzing text using machine learning...',
            'ipc_codes': ['G06F 40/30', 'G06F 17/27'],
            'cpc_codes': ['G06F 40/30'],
            'keywords': ['nlp', 'machine learning', 'text analysis'],
            'keywords_formatted': 'nlp (Natural Language Processing); machine learning; text analysis',
            'citations': ['US 1234567', 'US 7654321'],
            'publication_year': '2024',
            'filename': 'patent1.pdf',
            'num_pages': 20,
            'summary': 'This invention relates to automated text processing.'
        }
    ]
    
    csv_path = generator.generate_csv(sample_results, 'test_output.csv')
    generator.print_summary(csv_path, sample_results)