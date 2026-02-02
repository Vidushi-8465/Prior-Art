"""
Enhanced Pipeline Demo
----------------------
Demo script showing how to use the new PDF-to-CSV patent analysis pipeline.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import PatentAnalysisPipeline


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def demo_single_pdf(pipeline, pdf_path):
    """Demo with a single PDF file."""
    print_header("DEMO: SINGLE PDF ANALYSIS")
    
    print(f"Analyzing: {pdf_path}\n")
    
    # Run the pipeline
    csv_path = pipeline.run_pipeline(
        pdf_paths=pdf_path,
        num_keywords=15,
        output_filename="single_patent_analysis.csv"
    )
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ CSV saved to: {csv_path}")


def demo_multiple_pdfs(pipeline, pdf_folder):
    """Demo with multiple PDF files from a folder."""
    print_header("DEMO: MULTIPLE PDF ANALYSIS")
    
    # Get all PDF files in folder
    pdf_files = []
    if os.path.exists(pdf_folder):
        for file in os.listdir(pdf_folder):
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(pdf_folder, file))
    
    if not pdf_files:
        print(f"No PDF files found in: {pdf_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Run the pipeline
    csv_path = pipeline.run_pipeline(
        pdf_paths=pdf_files,
        num_keywords=15,
        output_filename="batch_patent_analysis.csv"
    )
    
    print(f"\n✓ Batch analysis complete!")
    print(f"✓ CSV saved to: {csv_path}")


def interactive_mode():
    """Interactive mode - ask user for PDF path."""
    print_header("INTERACTIVE PATENT ANALYSIS")
    
    print("This tool will:")
    print("  1. Extract text and metadata from your PDF")
    print("  2. Generate a summary")
    print("  3. Extract keywords (with full forms like NLP → Natural Language Processing)")
    print("  4. Find IPC/CPC codes, citations, and publication year")
    print("  5. Save everything to a CSV file")
    print("\n" + "-"*70 + "\n")
    
    # Initialize pipeline
    pipeline = PatentAnalysisPipeline(output_dir="../data/output")
    
    # Get PDF path from user
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    # Remove quotes if user added them
    pdf_path = pdf_path.strip('"').strip("'")
    
    if not os.path.exists(pdf_path):
        print(f"\n✗ Error: File not found: {pdf_path}")
        return
    
    if not pdf_path.endswith('.pdf'):
        print(f"\n✗ Error: File must be a PDF")
        return
    
    # Ask for number of keywords
    try:
        num_kw = input("\nHow many keywords to extract? [default: 15]: ").strip()
        num_keywords = int(num_kw) if num_kw else 15
    except:
        num_keywords = 15
    
    print("\n" + "="*70)
    print("Starting analysis...")
    print("="*70)
    
    # Run analysis
    try:
        csv_path = pipeline.run_pipeline(
            pdf_paths=pdf_path,
            num_keywords=num_keywords
        )
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nYour CSV file is ready: {csv_path}")
        print("\nYou can now open it in Excel or any spreadsheet software.")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function."""
    print_header("PATENT ANALYSIS PIPELINE - DEMO")
    
    print("Choose a demo mode:\n")
    print("  1. Interactive mode (enter PDF path)")
    print("  2. Example: Single PDF (requires sample file)")
    print("  3. Example: Multiple PDFs (requires sample folder)")
    print("  4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        interactive_mode()
    
    elif choice == "2":
        pipeline = PatentAnalysisPipeline(output_dir="../data/output")
        
        # Ask for PDF path
        pdf_path = input("\nEnter path to PDF file: ").strip().strip('"').strip("'")
        
        if os.path.exists(pdf_path):
            demo_single_pdf(pipeline, pdf_path)
        else:
            print(f"File not found: {pdf_path}")
    
    elif choice == "3":
        pipeline = PatentAnalysisPipeline(output_dir="../data/output")
        
        # Ask for folder path
        folder_path = input("\nEnter path to folder containing PDFs: ").strip().strip('"').strip("'")
        
        if os.path.exists(folder_path):
            demo_multiple_pdfs(pipeline, folder_path)
        else:
            print(f"Folder not found: {folder_path}")
    
    elif choice == "4":
        print("\nGoodbye!")
        return
    
    else:
        print("\nInvalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()