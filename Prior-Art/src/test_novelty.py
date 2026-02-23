from novelty_ranker import NoveltyRanker
from PyPDF2 import PdfReader
import os

if __name__ == "__main__":
    print("\n==============================")
    print("   PATENT NOVELTY ANALYZER")
    print("==============================\n")

    # Ask user for PDF path instead of text input
    pdf_path = input("Enter the path to your PDF file: ").strip().strip('"')

    # Verify the file exists
    if not os.path.exists(pdf_path):
        print(f"\n✗ Error: File not found — {pdf_path}")
        exit()

    # Extract text from the PDF
    print("\nExtracting text from PDF...\n")
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print(f"✗ Error reading PDF: {e}")
        exit()

    # Use the extracted text as the invention abstract
    abstract = text.strip()
    print("✓ Text extracted successfully!\n")

    # Initialize ranker
    ranker = NoveltyRanker()
    
    # Compute novelty ranking (use abstract instead of undefined 'idea')
    results = ranker.rank_novelty(abstract, top_k=5)

    print("\n=== Top Similar Patents ===")
    for r in results:
        print(f"{r['rank']}. {r['title']}  |  Similarity: {r['similarity']:.3f}  |  Novelty: {r['novelty_score']}%")

    print("\n✅ Done! Results computed successfully.")