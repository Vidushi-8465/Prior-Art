"""
Enhanced PDF Extraction Module
-------------------------------
Extracts text and patent metadata (IPC, CPC codes, publication year, citations, etc.)
"""

import PyPDF2
import pdfplumber
import re
from typing import Dict, List, Optional
import os


class PatentPDFExtractor:
    """
    Extract text and patent-specific metadata from PDF files.
    """
    
    @staticmethod
    def extract_text_with_pdfplumber(pdf_path: str) -> str:
        """Extract text using pdfplumber."""
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
    def extract_publication_year(text: str, metadata: Dict) -> Optional[str]:
        """
        Extract publication year from PDF text or metadata.
        
        Args:
            text: PDF text content
            metadata: PDF metadata dictionary
            
        Returns:
            Publication year as string or None
        """
        # Try metadata first
        if metadata:
            creation_date = metadata.get('/CreationDate', '')
            if creation_date and len(creation_date) >= 4:
                year_match = re.search(r'(\d{4})', creation_date)
                if year_match:
                    return year_match.group(1)
        
        # Search in text for common patent year patterns
        patterns = [
            r'Publication\s+(?:Date|Year)[:\s]+(\d{4})',
            r'Filed[:\s]+.*?(\d{4})',
            r'Patent\s+No\..*?(\d{4})',
            r'Pub(?:lished|lication).*?(\d{4})',
            r'\b(19\d{2}|20\d{2})\b'  # Years between 1900-2099
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = match.group(1)
                # Validate year is reasonable (1900-2030)
                if 1900 <= int(year) <= 2030:
                    return year
        
        return None
    
    @staticmethod
    def extract_ipc_codes(text: str) -> List[str]:
        """
        Extract IPC (International Patent Classification) codes.
        
        Format examples:
        - G06F 40/30
        - H04L 29/06
        - A61B 5/00
        
        Args:
            text: PDF text content
            
        Returns:
            List of IPC codes
        """
        ipc_codes = set()
        
        # Pattern for IPC codes: Letter + 2 digits + Letter + space + numbers/numbers
        pattern = r'\b([A-H]\d{2}[A-Z]\s*\d+/\d+)\b'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Normalize format (remove extra spaces)
            code = re.sub(r'\s+', ' ', match.strip())
            ipc_codes.add(code)
        
        # Also look for "IPC:" or "Int. Cl.:" patterns
        ipc_section = re.search(r'(?:IPC|Int\.?\s*Cl\.?)[:\s]+([A-H0-9\s,;/]+)', text, re.IGNORECASE)
        if ipc_section:
            codes_text = ipc_section.group(1)
            additional_codes = re.findall(r'([A-H]\d{2}[A-Z]\s*\d+/\d+)', codes_text)
            for code in additional_codes:
                ipc_codes.add(re.sub(r'\s+', ' ', code.strip()))
        
        return sorted(list(ipc_codes))
    
    @staticmethod
    def extract_cpc_codes(text: str) -> List[str]:
        """
        Extract CPC (Cooperative Patent Classification) codes.
        
        Similar format to IPC but may have additional components.
        
        Args:
            text: PDF text content
            
        Returns:
            List of CPC codes
        """
        cpc_codes = set()
        
        # Pattern for CPC codes (similar to IPC)
        pattern = r'\b([A-HY]\d{2}[A-Z]\s*\d+/\d+(?:\s*\d+)?)\b'
        matches = re.findall(pattern, text)
        
        for match in matches:
            code = re.sub(r'\s+', ' ', match.strip())
            cpc_codes.add(code)
        
        # Look for "CPC:" pattern
        cpc_section = re.search(r'CPC[:\s]+([A-HY0-9\s,;/]+)', text, re.IGNORECASE)
        if cpc_section:
            codes_text = cpc_section.group(1)
            additional_codes = re.findall(r'([A-HY]\d{2}[A-Z]\s*\d+/\d+(?:\s*\d+)?)', codes_text)
            for code in additional_codes:
                cpc_codes.add(re.sub(r'\s+', ' ', code.strip()))
        
        return sorted(list(cpc_codes))
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """
        Extract patent citations and references.
        
        Looks for:
        - Patent numbers (US 1234567, EP 0123456, etc.)
        - References cited section
        
        Args:
            text: PDF text content
            
        Returns:
            List of citation references
        """
        citations = set()
        
        # Pattern for US patent numbers
        us_patents = re.findall(r'\b(US\s*\d{7,8}(?:\s*[A-Z]\d?)?)\b', text, re.IGNORECASE)
        for patent in us_patents:
            citations.add(re.sub(r'\s+', ' ', patent.strip().upper()))
        
        # Pattern for EP patent numbers
        ep_patents = re.findall(r'\b(EP\s*\d{7}(?:\s*[A-Z]\d?)?)\b', text, re.IGNORECASE)
        for patent in ep_patents:
            citations.add(re.sub(r'\s+', ' ', patent.strip().upper()))
        
        # Pattern for WO (WIPO) patent numbers
        wo_patents = re.findall(r'\b(WO\s*\d{4}/\d{6})\b', text, re.IGNORECASE)
        for patent in wo_patents:
            citations.add(re.sub(r'\s+', ' ', patent.strip().upper()))
        
        # Look in "References Cited" section
        ref_section = re.search(r'References?\s+Cited[:\s]+(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if ref_section:
            ref_text = ref_section.group(1)
            # Extract patent numbers from references section
            more_patents = re.findall(r'\b((?:US|EP|WO)\s*[\d/]+(?:\s*[A-Z]\d?)?)\b', ref_text, re.IGNORECASE)
            for patent in more_patents:
                citations.add(re.sub(r'\s+', ' ', patent.strip().upper()))
        
        return sorted(list(citations))[:20]  # Limit to top 20
    
    @staticmethod
    def extract_title(text: str) -> Optional[str]:
        """
        Extract patent/document title.
        
        Args:
            text: PDF text content
            
        Returns:
            Title string or None
        """
        # First 500 chars usually contain title
        header = text[:500]
        
        # Look for common title patterns
        patterns = [
            r'Title[:\s]+(.+?)(?:\n|$)',
            r'TITLE[:\s]+(.+?)(?:\n|$)',
            r'^\s*(.+?)\n',  # First line
        ]
        
        for pattern in patterns:
            match = re.search(pattern, header, re.MULTILINE)
            if match:
                title = match.group(1).strip()
                # Clean up
                title = re.sub(r'\s+', ' ', title)
                if 10 < len(title) < 200:  # Reasonable title length
                    return title
        
        return None
    
    @staticmethod
    def extract_abstract(text: str) -> Optional[str]:
        """
        Extract abstract from patent document.
        
        Args:
            text: PDF text content
            
        Returns:
            Abstract text or None
        """
        # Look for abstract section
        patterns = [
            r'ABSTRACT[:\s]+(.*?)(?:BACKGROUND|FIELD|SUMMARY|BRIEF|DETAILED|\n\n\n)',
            r'Abstract[:\s]+(.*?)(?:\n\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r'\s+', ' ', abstract)
                abstract = re.sub(r'\n+', ' ', abstract)
                if 50 < len(abstract) < 2000:  # Reasonable abstract length
                    return abstract
        
        return None
    
    @staticmethod
    def extract_all(pdf_path: str) -> Dict:
        """
        Extract all information from patent PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with all extracted information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text
        text = PatentPDFExtractor.extract_text_with_pdfplumber(pdf_path)
        
        # Get basic metadata
        metadata = {}
        num_pages = 0
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata if pdf_reader.metadata else {}
                num_pages = len(pdf_reader.pages)
        except:
            pass
        
        # Extract all patent-specific information
        result = {
            'filename': os.path.basename(pdf_path),
            'filepath': pdf_path,
            'num_pages': num_pages,
            'text': text,
            'title': PatentPDFExtractor.extract_title(text),
            'abstract': PatentPDFExtractor.extract_abstract(text),
            'publication_year': PatentPDFExtractor.extract_publication_year(text, metadata),
            'ipc_codes': PatentPDFExtractor.extract_ipc_codes(text),
            'cpc_codes': PatentPDFExtractor.extract_cpc_codes(text),
            'citations': PatentPDFExtractor.extract_citations(text),
            'metadata': metadata
        }
        
        return result


if __name__ == "__main__":
    # Example usage
    extractor = PatentPDFExtractor()
    
    # Test with a sample patent PDF
    # result = extractor.extract_all("sample_patent.pdf")
    # print(f"Title: {result['title']}")
    # print(f"Year: {result['publication_year']}")
    # print(f"IPC Codes: {result['ipc_codes']}")
    # print(f"CPC Codes: {result['cpc_codes']}")
    # print(f"Citations: {result['citations'][:5]}")
    
    print("PatentPDFExtractor ready!")