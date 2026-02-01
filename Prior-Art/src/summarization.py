"""
Summarization Module
--------------------
Provides text summarization using sumy's TextRank algorithm.
"""

from typing import Optional
import warnings
import re
warnings.filterwarnings('ignore')


class TextSummarizer:
    """
    Simple summarizer using sentence extraction.
    """
    
    @staticmethod
    def simple_summarize(text: str, num_sentences: int = 3) -> str:
        """
        Simple sentence extraction based summarization.
        Useful when other methods fail on short texts.
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            Summary with top N sentences
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If fewer sentences than requested, return all
        if len(sentences) <= num_sentences:
            return text
        
        # Return first N sentences (simple but effective)
        return '. '.join(sentences[:num_sentences]) + '.'


class ModernTextSummarizer:
    """
    Modern summarizer using sumy (compatible with latest libraries).
    """
    
    @staticmethod
    def summarize_with_textrank(text: str, sentence_count: int = 3) -> str:
        """
        Summarize using TextRank algorithm via sumy.
        
        Args:
            text: Input text
            sentence_count: Number of sentences in summary
            
        Returns:
            Summarized text
        """
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, sentence_count)
            
            return ' '.join([str(sentence) for sentence in summary])
        except ImportError:
            print("sumy not installed. Using simple summarization...")
            return TextSummarizer.simple_summarize(text, sentence_count)
        except Exception as e:
            print(f"Error in summarization: {e}. Using simple summarization...")
            return TextSummarizer.simple_summarize(text, sentence_count)
    
    @staticmethod
    def summarize(text: str, 
                  method: str = "textrank",
                  ratio: float = 0.3,
                  num_sentences: int = 3) -> str:
        """
        Main summarization method with fallback.
        
        Args:
            text: Input text
            method: 'textrank' or 'simple'
            ratio: Ratio for summarization (not used in simple mode)
            num_sentences: Number of sentences for summary
            
        Returns:
            Summarized text
        """
        if method == "textrank":
            return ModernTextSummarizer.summarize_with_textrank(text, num_sentences)
        else:
            return TextSummarizer.simple_summarize(text, num_sentences)


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and 
    human language. It involves programming computers to process and analyze large 
    amounts of natural language data. The goal is a computer capable of understanding 
    the contents of documents, including the contextual nuances of the language within them. 
    The technology can then accurately extract information and insights contained in the 
    documents as well as categorize and organize the documents themselves. Challenges in 
    natural language processing frequently involve speech recognition, natural language 
    understanding, and natural language generation.
    """
    
    print("Original Text:")
    print(sample_text)
    print(f"\nLength: {len(sample_text)} characters\n")
    print("="*70 + "\n")
    
    print("Simple Summary (3 sentences):")
    simple_summary = TextSummarizer.simple_summarize(sample_text, num_sentences=2)
    print(simple_summary)
    print(f"\nLength: {len(simple_summary)} characters\n")
    print("="*70 + "\n")
    
    print("TextRank Summary:")
    modern_summary = ModernTextSummarizer.summarize_with_textrank(sample_text, sentence_count=2)
    print(modern_summary)
    print(f"\nLength: {len(modern_summary)} characters")