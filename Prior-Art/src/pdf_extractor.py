"""
PDF Extraction Module
---------------------
Handles PDF file reading and text extraction.
"""

import PyPDF2
import pdfplumber
from typing import Optional, Dict
import os


class PDFExtractor:
    """
    Extract text from PDF files using multiple methods.
    """
    
    @staticmethod
    def extract_with_pypdf2(pdf_path: str) -> str:
        """
        Extract text using PyPDF2 (faster but less accurate).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting with PyPDF2: {e}")
        
        return text.strip()
    
    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> str:
        """
        Extract text using pdfplumber (slower but more accurate).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting with pdfplumber: {e}")
        
        return text.strip()
    
    @staticmethod
    def extract(pdf_path: str, method: str = "pdfplumber") -> Dict[str, str]:
        """
        Main extraction method with metadata.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pypdf2' or 'pdfplumber')
            
        Returns:
            Dictionary with text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text
        if method.lower() == "pypdf2":
            text = PDFExtractor.extract_with_pypdf2(pdf_path)
        else:
            text = PDFExtractor.extract_with_pdfplumber(pdf_path)
        
        # Get metadata
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                num_pages = len(pdf_reader.pages)
        except:
            metadata = {}
            num_pages = 0
        
        return {
            'text': text,
            'filename': os.path.basename(pdf_path),
            'num_pages': num_pages,
            'metadata': metadata
        }
    
    @staticmethod
    def extract_from_text_input(text: str) -> Dict[str, str]:
        """
        Handle direct text input (when user provides text instead of PDF).
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with text and metadata
        """
        return {
            'text': text,
            'filename': 'direct_input.txt',
            'num_pages': 1,
            'metadata': {}
        }


if __name__ == "__main__":
    # Example usage
    extractor = PDFExtractor()
    
    # For text input
    sample_text = """
    A system and method for processing natural language using deep learning.
    The invention comprises a novel architecture for understanding context.
    """
    
    result = extractor.extract_from_text_input(sample_text)
    print("Extracted from text input:")
    print(f"Text length: {len(result['text'])} characters")
    print(f"First 100 chars: {result['text'][:100]}")