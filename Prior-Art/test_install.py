print("Testing imports...")

try:
    import spacy
    print("✓ spaCy installed")
except:
    print("✗ spaCy failed")

try:
    import sklearn
    print("✓ scikit-learn installed")
except:
    print("✗ scikit-learn failed")

try:
    import yake
    print("✓ YAKE installed")
except:
    print("✗ YAKE failed")

try:
    import rake_nltk
    print("✓ RAKE installed")
except:
    print("✗ RAKE failed")

try:
    from keybert import KeyBERT
    print("✓ KeyBERT installed")
except:
    print("⚠ KeyBERT not installed (optional)")

try:
    import gensim
    print("✓ Gensim installed")
except:
    print("✗ Gensim failed")
try:
    import sumy
    print("✓ sumy installed")
except:
    print("✗ sumy failed")

try:
    import PyPDF2
    print("✓ PyPDF2 installed")
except:
    print("✗ PyPDF2 failed")

print("\nDone!")