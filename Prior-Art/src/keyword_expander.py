"""
Keyword Expansion Module
-------------------------
Expands keywords with synonyms, full forms, and related terms.
"""

from typing import List, Dict, Tuple
import re


class KeywordExpander:
    """
    Expand keywords with synonyms and full forms (e.g., NLP -> Natural Language Processing).
    """
    
    # Common acronyms and their expansions (tech/patent focused)
    ACRONYM_DATABASE = {
        # AI & ML
        'ai': 'Artificial Intelligence',
        'ml': 'Machine Learning',
        'dl': 'Deep Learning',
        'nn': 'Neural Network',
        'cnn': 'Convolutional Neural Network',
        'rnn': 'Recurrent Neural Network',
        'lstm': 'Long Short-Term Memory',
        'gru': 'Gated Recurrent Unit',
        'gan': 'Generative Adversarial Network',
        'nlp': 'Natural Language Processing',
        'nlu': 'Natural Language Understanding',
        'nlg': 'Natural Language Generation',
        'cv': 'Computer Vision',
        'ocr': 'Optical Character Recognition',
        'asr': 'Automatic Speech Recognition',
        'tts': 'Text-to-Speech',
        'rl': 'Reinforcement Learning',
        'svm': 'Support Vector Machine',
        'knn': 'K-Nearest Neighbors',
        'pca': 'Principal Component Analysis',
        'ann': 'Artificial Neural Network',
        'mlp': 'Multi-Layer Perceptron',
        
        # Data & Computing
        'api': 'Application Programming Interface',
        'sdk': 'Software Development Kit',
        'ui': 'User Interface',
        'ux': 'User Experience',
        'gui': 'Graphical User Interface',
        'cli': 'Command Line Interface',
        'ide': 'Integrated Development Environment',
        'db': 'Database',
        'dbms': 'Database Management System',
        'sql': 'Structured Query Language',
        'nosql': 'Not Only SQL',
        'json': 'JavaScript Object Notation',
        'xml': 'Extensible Markup Language',
        'html': 'HyperText Markup Language',
        'css': 'Cascading Style Sheets',
        'http': 'HyperText Transfer Protocol',
        'https': 'HyperText Transfer Protocol Secure',
        'ftp': 'File Transfer Protocol',
        'ssh': 'Secure Shell',
        'vpn': 'Virtual Private Network',
        'lan': 'Local Area Network',
        'wan': 'Wide Area Network',
        'ip': 'Internet Protocol',
        'tcp': 'Transmission Control Protocol',
        'udp': 'User Datagram Protocol',
        'dns': 'Domain Name System',
        'url': 'Uniform Resource Locator',
        'uri': 'Uniform Resource Identifier',
        'rest': 'Representational State Transfer',
        'soap': 'Simple Object Access Protocol',
        
        # IoT & Hardware
        'iot': 'Internet of Things',
        'iiot': 'Industrial Internet of Things',
        'rfid': 'Radio-Frequency Identification',
        'nfc': 'Near Field Communication',
        'gps': 'Global Positioning System',
        'cpu': 'Central Processing Unit',
        'gpu': 'Graphics Processing Unit',
        'tpu': 'Tensor Processing Unit',
        'ram': 'Random Access Memory',
        'rom': 'Read-Only Memory',
        'ssd': 'Solid State Drive',
        'hdd': 'Hard Disk Drive',
        'usb': 'Universal Serial Bus',
        'hdmi': 'High-Definition Multimedia Interface',
        'led': 'Light Emitting Diode',
        'oled': 'Organic Light Emitting Diode',
        'lcd': 'Liquid Crystal Display',
        
        # Medical & Bio
        'ecg': 'Electrocardiogram',
        'eeg': 'Electroencephalogram',
        'mri': 'Magnetic Resonance Imaging',
        'ct': 'Computed Tomography',
        'pet': 'Positron Emission Tomography',
        'dna': 'Deoxyribonucleic Acid',
        'rna': 'Ribonucleic Acid',
        'pcr': 'Polymerase Chain Reaction',
        'emr': 'Electronic Medical Record',
        'ehr': 'Electronic Health Record',
        'fda': 'Food and Drug Administration',
        
        # Business & Standards
        'roi': 'Return on Investment',
        'kpi': 'Key Performance Indicator',
        'crm': 'Customer Relationship Management',
        'erp': 'Enterprise Resource Planning',
        'saas': 'Software as a Service',
        'paas': 'Platform as a Service',
        'iaas': 'Infrastructure as a Service',
        'b2b': 'Business to Business',
        'b2c': 'Business to Consumer',
        'iso': 'International Organization for Standardization',
        'ieee': 'Institute of Electrical and Electronics Engineers',
        'ansi': 'American National Standards Institute',
        
        # Security & Crypto
        'ssl': 'Secure Sockets Layer',
        'tls': 'Transport Layer Security',
        'aes': 'Advanced Encryption Standard',
        'rsa': 'Rivest-Shamir-Adleman',
        'sha': 'Secure Hash Algorithm',
        'pki': 'Public Key Infrastructure',
        '2fa': 'Two-Factor Authentication',
        'mfa': 'Multi-Factor Authentication',
        'ddos': 'Distributed Denial of Service',
        
        # Other
        'pdf': 'Portable Document Format',
        'jpeg': 'Joint Photographic Experts Group',
        'png': 'Portable Network Graphics',
        'gif': 'Graphics Interchange Format',
        'svg': 'Scalable Vector Graphics',
        'mp3': 'MPEG Audio Layer 3',
        'mp4': 'MPEG-4 Part 14',
        'avi': 'Audio Video Interleave',
        'csv': 'Comma-Separated Values',
        'zip': 'Zone Improvement Plan',
        'ar': 'Augmented Reality',
        'vr': 'Virtual Reality',
        'xr': 'Extended Reality',
        '3d': 'Three-Dimensional',
        '2d': 'Two-Dimensional',
        'hd': 'High Definition',
        '4k': '4000 pixels',
        '5g': 'Fifth Generation',
        'lte': 'Long-Term Evolution',
        'wifi': 'Wireless Fidelity',
        'bluetooth': 'Bluetooth Wireless Technology',
        'os': 'Operating System',
        'bios': 'Basic Input/Output System',
        'dos': 'Disk Operating System',
        'pos': 'Point of Sale',
        'atm': 'Automated Teller Machine',
        'qr': 'Quick Response',
        'ocr': 'Optical Character Recognition',
    }
    
    @staticmethod
    def expand_acronym(keyword: str) -> List[str]:
        """
        Expand an acronym to its full form.
        
        Args:
            keyword: The keyword (possibly an acronym)
            
        Returns:
            List of expanded forms (includes original if not found)
        """
        keyword_lower = keyword.lower().strip()
        
        # Remove special characters for matching
        clean_key = re.sub(r'[^a-z0-9]', '', keyword_lower)
        
        expansions = [keyword]  # Always include original
        
        # Check if it's in our database
        if clean_key in KeywordExpander.ACRONYM_DATABASE:
            full_form = KeywordExpander.ACRONYM_DATABASE[clean_key]
            if full_form not in expansions:
                expansions.append(full_form)
        
        return expansions
    
    @staticmethod
    def detect_acronyms_in_text(text: str) -> Dict[str, str]:
        """
        Detect acronyms and their definitions in the text itself.
        
        Looks for patterns like:
        - "Natural Language Processing (NLP)"
        - "NLP (Natural Language Processing)"
        - "Machine Learning, or ML,"
        
        Args:
            text: Full text content
            
        Returns:
            Dictionary mapping acronyms to their full forms
        """
        found_acronyms = {}
        
        # Pattern 1: Full form (ACRONYM)
        pattern1 = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(([A-Z]{2,})\)'
        matches1 = re.findall(pattern1, text)
        for full_form, acronym in matches1:
            found_acronyms[acronym.lower()] = full_form
        
        # Pattern 2: ACRONYM (Full form)
        pattern2 = r'([A-Z]{2,})\s*\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\)'
        matches2 = re.findall(pattern2, text)
        for acronym, full_form in matches2:
            found_acronyms[acronym.lower()] = full_form
        
        # Pattern 3: "Full Form, or ACRONYM,"
        pattern3 = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s+or\s+([A-Z]{2,})'
        matches3 = re.findall(pattern3, text)
        for full_form, acronym in matches3:
            found_acronyms[acronym.lower()] = full_form
        
        return found_acronyms
    
    @staticmethod
    def expand_keywords(keywords: List[str], full_text: str = "") -> List[Tuple[str, List[str]]]:
        """
        Expand a list of keywords with their full forms and synonyms.
        
        Args:
            keywords: List of keywords to expand
            full_text: Optional full text to detect acronyms from
            
        Returns:
            List of tuples (original_keyword, [expansions])
        """
        # First, detect acronyms from the text itself
        text_acronyms = {}
        if full_text:
            text_acronyms = KeywordExpander.detect_acronyms_in_text(full_text)
        
        expanded_keywords = []
        
        for keyword in keywords:
            keyword_clean = keyword.lower().strip()
            expansions = [keyword]  # Always include original
            
            # Check text-detected acronyms first (more accurate)
            if keyword_clean in text_acronyms:
                full_form = text_acronyms[keyword_clean]
                if full_form not in expansions:
                    expansions.append(full_form)
            
            # Then check our database
            database_expansions = KeywordExpander.expand_acronym(keyword)
            for exp in database_expansions:
                if exp not in expansions and exp != keyword:
                    expansions.append(exp)
            
            expanded_keywords.append((keyword, expansions))
        
        return expanded_keywords
    
    @staticmethod
    def format_expanded_keywords(expanded: List[Tuple[str, List[str]]]) -> str:
        """
        Format expanded keywords as a readable string.
        
        Args:
            expanded: List of (keyword, expansions) tuples
            
        Returns:
            Formatted string
        """
        result = []
        for keyword, expansions in expanded:
            if len(expansions) > 1:
                # Has expansions
                others = [e for e in expansions if e != keyword]
                result.append(f"{keyword} ({', '.join(others)})")
            else:
                result.append(keyword)
        
        return "; ".join(result)


if __name__ == "__main__":
    # Test
    expander = KeywordExpander()
    
    # Test acronym expansion
    test_keywords = ['nlp', 'machine learning', 'cnn', 'api', 'iot']
    
    sample_text = """
    Natural Language Processing (NLP) is a field of AI. 
    We use Convolutional Neural Networks (CNN) for image processing.
    """
    
    expanded = expander.expand_keywords(test_keywords, sample_text)
    
    print("Expanded Keywords:")
    for keyword, expansions in expanded:
        print(f"  {keyword} -> {expansions}")
    
    print("\nFormatted:")
    print(expander.format_expanded_keywords(expanded))