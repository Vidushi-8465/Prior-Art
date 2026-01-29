"""
Quick Demo Script
-----------------
Run this script to see the prior art search system in action!

Usage:
    cd src
    python demo.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import PriorArtPipeline

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def main():
    """Run a complete demo of the prior art search system."""
    
    print_header("PRIOR ART SEARCH SYSTEM - DEMO")
    
    print("This demo will:")
    print("  1. Analyze a sample invention")
    print("  2. Extract keywords")
    print("  3. Compare with prior art")
    print("  4. Generate novelty assessment")
    print("\nPlease wait while we initialize the system...\n")
    
    # Initialize pipeline
    pipeline = PriorArtPipeline(output_dir="../data/output")
    
    # Sample invention: AI-powered medical diagnosis
    invention_description = """
    A novel artificial intelligence system for automated medical diagnosis using deep learning 
    and computer vision. The invention comprises a convolutional neural network architecture 
    that analyzes medical images (X-rays, CT scans, MRIs) to detect abnormalities with 98% 
    accuracy. The system includes a multi-stage pipeline: image preprocessing, feature extraction 
    using transfer learning from pre-trained models, and classification using ensemble methods. 
    The invention also includes an explainability module that highlights regions of interest 
    in the medical images, helping radiologists understand the AI's decision-making process. 
    Applications include early detection of tumors, fractures, and cardiovascular diseases.
    """
    
    # Sample prior art database
    prior_art_database = [
        {
            'text': 'A deep learning system for medical image classification using convolutional neural networks to identify pathologies in radiological images.',
            'metadata': {'id': 'PA001', 'year': 2019, 'source': 'Patent Database'}
        },
        {
            'text': 'Methods for tumor detection in radiological images using machine learning algorithms and pattern recognition techniques.',
            'metadata': {'id': 'PA002', 'year': 2020, 'source': 'Patent Database'}
        },
        {
            'text': 'Computer-aided diagnosis system for analyzing chest X-rays with artificial intelligence to detect pulmonary diseases.',
            'metadata': {'id': 'PA003', 'year': 2021, 'source': 'Patent Database'}
        },
        {
            'text': 'Explainable AI framework for medical image analysis providing interpretable decision support for clinical diagnosis.',
            'metadata': {'id': 'PA004', 'year': 2022, 'source': 'Patent Database'}
        },
        {
            'text': 'Automated detection of cardiovascular diseases using deep neural networks trained on cardiac imaging data.',
            'metadata': {'id': 'PA005', 'year': 2021, 'source': 'Patent Database'}
        },
        {
            'text': 'Image processing system for enhancing quality of medical scans using digital filtering and enhancement algorithms.',
            'metadata': {'id': 'PA006', 'year': 2018, 'source': 'Patent Database'}
        },
        {
            'text': 'Transfer learning approaches for medical image classification tasks using pre-trained convolutional neural networks.',
            'metadata': {'id': 'PA007', 'year': 2020, 'source': 'Patent Database'}
        },
        {
            'text': 'Ensemble methods for improving accuracy in diagnostic AI systems combining multiple machine learning models.',
            'metadata': {'id': 'PA008', 'year': 2021, 'source': 'Patent Database'}
        }
    ]
    
    print_header("SAMPLE INVENTION")
    print(invention_description.strip())
    
    print_header("ANALYZING...")
    
    # Run the complete pipeline
    # Using TF-IDF for faster demo (change to 'hybrid' for better accuracy)
    results = pipeline.run_full_pipeline(
        invention_input=invention_description,
        prior_art_docs=prior_art_database,
        is_file=False,
        similarity_method="tfidf",
        save_results=True
    )
    
    # Display results
    pipeline.print_results(results)
    
    # Additional insights
    print_header("RECOMMENDATIONS")
    
    novelty_score = results['prior_art_comparison']['novelty_metrics']['novelty_score']
    
    if novelty_score > 0.7:
        print("✓ HIGH NOVELTY DETECTED")
        print("\nRecommendations:")
        print("  • Proceed with detailed patent application")
        print("  • Emphasize the unique aspects in claims")
        print("  • Consider filing provisional patent")
        
    elif novelty_score > 0.4:
        print("⚠ MODERATE NOVELTY DETECTED")
        print("\nRecommendations:")
        print("  • Review the most similar prior art carefully")
        print("  • Identify and emphasize distinguishing features")
        print("  • Consider narrower claims or additional innovations")
        print("  • Consult with patent attorney")
        
    else:
        print("✗ LOW NOVELTY DETECTED")
        print("\nRecommendations:")
        print("  • Significant prior art exists")
        print("  • Consider major modifications to the invention")
        print("  • Explore alternative approaches")
        print("  • Focus on specific unique implementations")
    
    print("\n" + "="*70)
    print("\nDemo completed! Results saved to: data/output/")
    print("\nNext steps:")
    print("  • Try the web interface: python web_interface.py")
    print("  • Explore the Jupyter notebook: jupyter notebook")
    print("  • Modify this script to analyze your own invention")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print("\nPlease make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("  python -m spacy download en_core_web_sm")