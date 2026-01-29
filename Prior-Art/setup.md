# Step-by-Step Guide: Prior Art Search System

## 🎯 Quick Start Guide

### Step 1: System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection (for initial model downloads)
- 2GB free disk space

### Step 2: Installation

#### Windows
```bash
# Open Command Prompt or PowerShell

# Navigate to project directory
cd path\to\prior_art_search

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

#### Mac/Linux
```bash
# Open Terminal

# Navigate to project directory
cd path/to/prior_art_search

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 3: Verify Installation

```bash
cd src
python preprocessing.py
```

If you see output without errors, installation is successful!

---

## 📖 Usage Examples

### Example 1: Simple Command Line Usage

Create a file `my_analysis.py`:

```python
from src.pipeline import PriorArtPipeline

# Initialize
pipeline = PriorArtPipeline()

# Your invention
invention = """
Your invention description here.
This can be 2-3 paragraphs explaining your novel idea.
"""

# Prior art (from your research)
prior_art = [
    {'text': 'First prior art document...', 'metadata': {'id': 'P1'}},
    {'text': 'Second prior art document...', 'metadata': {'id': 'P2'}},
]

# Run analysis
results = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=prior_art,
    is_file=False,
    similarity_method="hybrid"
)

# Print results
pipeline.print_results(results)
```

Run it:
```bash
python my_analysis.py
```

### Example 2: Web Interface

```bash
cd src
python web_interface.py
```

Then:
1. Open browser to `http://localhost:5000`
2. Paste your invention description
3. Add prior art documents (one per line)
4. Click "Analyze"

### Example 3: Jupyter Notebook

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open notebooks/prior_art_analysis.ipynb
```

### Example 4: Using Individual Modules

#### Just Keyword Extraction
```python
from src.keyword_extraction import KeywordExtractor

extractor = KeywordExtractor()
text = "Your text here..."

# Using YAKE (recommended for beginners)
keywords = extractor.extract_with_yake(text, top_n=10)
for kw, score in keywords:
    print(f"{kw}: {score}")
```

#### Just Similarity Comparison
```python
from src.similarity_ranking import CitationRanker

ranker = CitationRanker()

invention = "Your invention..."
prior_art_docs = [
    {'text': 'Prior art 1...', 'metadata': {'id': '1'}},
    {'text': 'Prior art 2...', 'metadata': {'id': '2'}},
]

ranked = ranker.rank_documents(
    query_text=invention,
    documents=prior_art_docs,
    method="tfidf",
    top_n=5
)

for rank, doc, score in ranked:
    print(f"Rank {rank}: {score:.2%} - {doc['metadata']['id']}")
```

---

## 🔧 Detailed Workflow

### Workflow 1: Analyzing a New Invention

**Input**: Your invention description (text or PDF)

```python
from src.pipeline import PriorArtPipeline

pipeline = PriorArtPipeline()

# Option A: Text input
invention_text = "Your invention description..."
results = pipeline.run_full_pipeline(
    invention_input=invention_text,
    prior_art_docs=prior_art_list,
    is_file=False
)

# Option B: PDF input
results = pipeline.run_full_pipeline(
    invention_input="path/to/invention.pdf",
    prior_art_docs=prior_art_list,
    is_file=True
)
```

**Output**: JSON file with:
- Summary
- Keywords
- Novelty score
- Ranked prior art

### Workflow 2: Building a Prior Art Database

```python
# Step 1: Collect prior art
prior_art_collection = []

# From PDFs
from src.pdf_extractor import PDFExtractor
extractor = PDFExtractor()

for pdf_file in ['patent1.pdf', 'patent2.pdf']:
    result = extractor.extract(pdf_file)
    prior_art_collection.append({
        'text': result['text'],
        'metadata': {'filename': pdf_file}
    })

# Step 2: Analyze against your invention
results = pipeline.run_full_pipeline(
    invention_input=your_invention,
    prior_art_docs=prior_art_collection,
    similarity_method="hybrid"
)
```

### Workflow 3: Batch Processing

```python
# Process multiple inventions
inventions = [
    "Invention 1 description...",
    "Invention 2 description...",
    "Invention 3 description..."
]

all_results = []
for i, inv in enumerate(inventions, 1):
    print(f"Processing invention {i}...")
    result = pipeline.run_full_pipeline(
        invention_input=inv,
        prior_art_docs=prior_art_database,
        save_results=True
    )
    all_results.append(result)
```

---

## 📊 Understanding Results

### Novelty Score Interpretation

| Score Range | Interpretation | Recommendation |
|------------|----------------|----------------|
| 0.7 - 1.0 | **HIGH Novelty** | Strong candidate for patent filing |
| 0.4 - 0.7 | **MODERATE Novelty** | Review similarities, emphasize unique aspects |
| 0.0 - 0.4 | **LOW Novelty** | Significant prior art exists, consider modifications |

### Similarity Methods Comparison

**TF-IDF** (Term Frequency-Inverse Document Frequency)
- ✓ Fast
- ✓ Good for keyword matching
- ✗ Misses semantic relationships
- **Use when**: Speed is priority, large corpus

**BERT** (Bidirectional Encoder Representations)
- ✓ Understands context and semantics
- ✓ Best accuracy
- ✗ Slower, needs more resources
- **Use when**: Accuracy is priority, smaller corpus

**Hybrid** (Combines TF-IDF + BERT)
- ✓ Best of both worlds
- ✓ Balanced speed and accuracy
- **Use when**: General purpose (recommended)

---

## 🎨 Customization

### Adjusting Summary Length

```python
from src.summarization import ModernTextSummarizer

summarizer = ModernTextSummarizer()

# Short summary (2 sentences)
short = summarizer.summarize_with_textrank(text, sentence_count=2)

# Medium summary (5 sentences)
medium = summarizer.summarize_with_textrank(text, sentence_count=5)
```

### Changing Keyword Count

```python
from src.keyword_extraction import KeywordExtractor

extractor = KeywordExtractor()

# Get 5 keywords
keywords_5 = extractor.extract_with_yake(text, top_n=5)

# Get 20 keywords
keywords_20 = extractor.extract_with_yake(text, top_n=20)
```

### Using Different Keyword Methods

```python
# YAKE - Best for general use
yake_kw = extractor.extract_with_yake(text, top_n=10)

# RAKE - Best for speed
rake_kw = extractor.extract_with_rake(text, top_n=10)

# KeyBERT - Best for accuracy (slower)
keybert_kw = extractor.extract_with_keybert(text, top_n=10)

# Combined - Get unique keywords from all methods
combined = extractor.extract_combined(text, top_n=10)
unique = extractor.get_unique_keywords(combined, top_n=15)
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'spacy'"

**Solution**:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Problem: "OSError: [E050] Can't find model 'en_core_web_sm'"

**Solution**:
```bash
python -m spacy download en_core_web_sm
```

### Problem: KeyBERT is very slow

**Solutions**:
1. Use TF-IDF instead:
```python
results = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=prior_art,
    similarity_method="tfidf"  # Faster
)
```

2. Don't use KeyBERT for keyword extraction:
```python
# Use only YAKE
keywords = extractor.extract_with_yake(text, top_n=10)
```

### Problem: Out of memory errors

**Solutions**:
1. Process in smaller batches
2. Use TF-IDF instead of BERT
3. Reduce the number of prior art documents per batch

### Problem: PDF extraction returns empty text

**Solutions**:
1. Try different method:
```python
# Try pdfplumber
result = extractor.extract(pdf_path, method="pdfplumber")

# Or try PyPDF2
result = extractor.extract(pdf_path, method="pypdf2")
```

2. PDF might be image-based (needs OCR)

---

## 💡 Tips and Best Practices

### 1. Optimize for Your Use Case

**For Speed** (Processing many documents quickly):
```python
results = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=prior_art,
    similarity_method="tfidf",  # Fastest
    save_results=True
)
```

**For Accuracy** (Critical patent analysis):
```python
results = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=prior_art,
    similarity_method="hybrid",  # Best accuracy
    save_results=True
)
```

### 2. Prepare Good Prior Art

- Include diverse documents (both similar and different)
- Use complete descriptions (not just titles)
- Include metadata (publication date, ID, etc.)

### 3. Write Clear Invention Descriptions

- 2-3 paragraphs minimum
- Include technical details
- Mention key innovations
- Describe applications

### 4. Iterative Refinement

```python
# First pass - quick analysis
initial = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=all_prior_art,
    similarity_method="tfidf"
)

# Find top 20 most similar
top_20_similar = initial['prior_art_comparison']['ranked_citations'][:20]

# Second pass - detailed analysis on top results
detailed = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=[r[1] for r in top_20_similar],
    similarity_method="hybrid"
)
```

---

## 📚 Additional Resources

### Understanding the Algorithms

**YAKE**: Statistical keyword extraction
- Combines term frequency, casing, position
- No training required
- Language independent

**TF-IDF**: Term importance measurement
- Frequency in document vs corpus
- Fast and interpretable
- Good baseline method

**BERT**: Deep learning embeddings
- Understands context
- Pre-trained on large corpus
- Computationally intensive

### Extending the System

Want to add new features? Check out:
- `src/preprocessing.py` - Add custom text cleaning
- `src/keyword_extraction.py` - Add new keyword methods
- `src/similarity_ranking.py` - Add new similarity metrics
- `src/pipeline.py` - Modify the workflow

---

## 🎓 Learning Path

### Beginner (1-2 hours)
1. Run the web interface
2. Try with sample data
3. Understand the output

### Intermediate (3-5 hours)
1. Use command line scripts
2. Customize parameters
3. Process your own data

### Advanced (1-2 days)
1. Modify individual modules
2. Integrate with your database
3. Build custom workflows

---

## 📞 Getting Help

If you encounter issues:
1. Check this guide
2. Read module documentation
3. Review error messages carefully
4. Test with sample data first

---

**Happy Patent Searching! 🚀**