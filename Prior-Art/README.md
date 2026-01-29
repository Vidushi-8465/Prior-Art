# Prior Art Search System 🔍

A comprehensive NLP-based system for automated prior art search and novelty assessment in patent applications.

## 📋 Overview

This system provides an end-to-end pipeline for:
1. **Text Extraction**: Extract text from PDFs or accept direct input
2. **Preprocessing**: Clean and normalize text using spaCy
3. **Summarization**: Generate concise summaries using TextRank
4. **Keyword Extraction**: Extract relevant keywords using YAKE, RAKE, and KeyBERT
5. **Similarity Analysis**: Compare inventions with prior art using TF-IDF and BERT
6. **Citation Ranking**: Rank prior art by relevance
7. **Novelty Assessment**: Compute novelty scores

## 🏗️ Project Structure

```
prior_art_search/
├── src/
│   ├── preprocessing.py          # Text cleaning and preprocessing
│   ├── pdf_extractor.py          # PDF text extraction
│   ├── summarization.py          # Text summarization
│   ├── keyword_extraction.py     # Keyword extraction (YAKE, RAKE, KeyBERT)
│   ├── similarity_ranking.py     # Similarity computation and ranking
│   ├── pipeline.py               # Main pipeline integrating all components
│   └── web_interface.py          # Flask web interface
├── data/
│   ├── input/                    # Input files directory
│   └── output/                   # Output results directory
├── notebooks/                    # Jupyter notebooks for experimentation
├── tests/                        # Unit tests
└── requirements.txt              # Python dependencies
```

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
cd prior_art_search
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### Optional: Install sumy for Better Summarization

```bash
pip install sumy
```

## 📖 Usage

### Option 1: Command Line (Python Script)

```python
from src.pipeline import PriorArtPipeline

# Initialize pipeline
pipeline = PriorArtPipeline(output_dir="data/output")

# Your invention description
invention = """
A novel machine learning system for real-time emotion detection
using multimodal deep learning combining facial expressions and
voice patterns...
"""

# Prior art documents
prior_art = [
    {
        'text': 'Facial recognition system using CNNs...',
        'metadata': {'id': 'P001', 'year': 2020}
    },
    # Add more documents...
]

# Run analysis
results = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=prior_art,
    is_file=False,
    similarity_method="hybrid",
    save_results=True
)

# Print results
pipeline.print_results(results)
```

### Option 2: Web Interface

```bash
cd src
python web_interface.py
```

Then open your browser to `http://localhost:5000`

### Option 3: Individual Modules

#### Preprocessing
```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned = preprocessor.preprocess(text)
```

#### Keyword Extraction
```python
from src.keyword_extraction import KeywordExtractor

extractor = KeywordExtractor()
keywords = extractor.extract_with_yake(text, top_n=10)
```

#### Similarity Analysis
```python
from src.similarity_ranking import CitationRanker

ranker = CitationRanker()
ranked = ranker.rank_documents(query, documents, method="hybrid")
```

## 🔧 Module Details

### 1. Preprocessing (`preprocessing.py`)

**Features:**
- Text cleaning (remove URLs, emails, special characters)
- Lemmatization
- Stopword removal
- Noun phrase extraction

**Example:**
```python
preprocessor = TextPreprocessor()
clean_text = preprocessor.clean_text(raw_text)
lemmatized = preprocessor.lemmatize(text, remove_stopwords=True)
```

### 2. PDF Extraction (`pdf_extractor.py`)

**Features:**
- Extract text from PDFs using PyPDF2 or pdfplumber
- Handle both file and direct text input
- Extract metadata

**Example:**
```python
extractor = PDFExtractor()
result = extractor.extract("path/to/file.pdf", method="pdfplumber")
print(result['text'])
```

### 3. Summarization (`summarization.py`)

**Features:**
- TextRank-based extractive summarization
- Fallback to simple sentence extraction
- Configurable summary length

**Example:**
```python
summarizer = ModernTextSummarizer()
summary = summarizer.summarize_with_textrank(text, sentence_count=3)
```

### 4. Keyword Extraction (`keyword_extraction.py`)

**Algorithms:**
- **YAKE**: Statistical + linguistic features
- **RAKE**: Rapid keyword extraction
- **KeyBERT**: BERT-based semantic extraction

**Example:**
```python
extractor = KeywordExtractor()

# Single method
yake_kw = extractor.extract_with_yake(text, top_n=10)

# All methods combined
all_kw = extractor.extract_combined(text, top_n=10)
unique = extractor.get_unique_keywords(all_kw, top_n=15)
```

### 5. Similarity & Ranking (`similarity_ranking.py`)

**Methods:**
- **TF-IDF**: Fast, interpretable, statistical
- **BERT**: Semantic, context-aware
- **Hybrid**: Combines both (recommended)

**Example:**
```python
ranker = CitationRanker()

# Rank documents
ranked = ranker.rank_documents(
    query_text=invention,
    documents=prior_art_list,
    method="hybrid",
    top_n=10
)

# Compute novelty
novelty = ranker.compute_novelty_score(
    query_text=invention,
    prior_art_docs=prior_art_texts,
    method="hybrid"
)
```

### 6. Main Pipeline (`pipeline.py`)

**Complete Workflow:**
```python
pipeline = PriorArtPipeline()

results = pipeline.run_full_pipeline(
    invention_input=text_or_pdf_path,
    prior_art_docs=prior_art_list,
    is_file=False,
    similarity_method="hybrid",
    save_results=True
)
```

## 📊 Output Format

The pipeline generates a comprehensive JSON report:

```json
{
  "input_metadata": {
    "filename": "invention.txt",
    "num_pages": 1,
    "input_length": 1250
  },
  "analysis": {
    "original_text": "...",
    "cleaned_text": "...",
    "summary": "...",
    "keywords": {
      "yake": [...],
      "rake": [...],
      "unique_keywords": [...]
    }
  },
  "prior_art_comparison": {
    "ranked_citations": [...],
    "novelty_metrics": {
      "novelty_score": 0.72,
      "max_similarity": 0.28,
      "avg_similarity": 0.15
    }
  }
}
```

## 🎯 Use Cases

1. **Patent Filing**: Assess novelty before filing
2. **Prior Art Search**: Find relevant existing patents
3. **R&D**: Identify gaps in existing technology
4. **Legal Analysis**: Support patent litigation
5. **Technology Scouting**: Discover similar innovations

## ⚙️ Configuration

### Similarity Methods

- **TF-IDF**: Fastest, good for keyword matching
- **BERT**: Best semantic understanding, slower
- **Hybrid** (recommended): Balance of speed and accuracy

### Keyword Extraction

- **YAKE**: Best for general use, no training needed
- **RAKE**: Fast, good for technical documents
- **KeyBERT**: Most accurate, requires more resources

## 📈 Performance Tips

1. **For faster processing**: Use `tfidf` similarity method
2. **For better accuracy**: Use `hybrid` or `bert` method
3. **Memory optimization**: Avoid loading KeyBERT unless needed
4. **Large corpora**: Process in batches

## 🧪 Testing

Run individual module tests:

```bash
cd src
python preprocessing.py
python keyword_extraction.py
python similarity_ranking.py
python pipeline.py
```

## 🔍 Example Workflow

```python
# Step 1: Initialize
from src.pipeline import PriorArtPipeline
pipeline = PriorArtPipeline()

# Step 2: Prepare input
invention = "A neural network system for automated patent analysis..."

prior_art = [
    {'text': 'Patent 1 description...', 'metadata': {'id': 'P1'}},
    {'text': 'Patent 2 description...', 'metadata': {'id': 'P2'}},
]

# Step 3: Run analysis
results = pipeline.run_full_pipeline(
    invention_input=invention,
    prior_art_docs=prior_art,
    similarity_method="hybrid"
)

# Step 4: View results
pipeline.print_results(results)

# Step 5: Check novelty
novelty_score = results['prior_art_comparison']['novelty_metrics']['novelty_score']
if novelty_score > 0.7:
    print("HIGH novelty - proceed with patent filing")
elif novelty_score > 0.4:
    print("MODERATE novelty - review similar patents")
else:
    print("LOW novelty - significant prior art exists")
```

## 📝 Notes

- **First Run**: May take longer as models are downloaded
- **GPU**: Not required but speeds up BERT operations
- **Memory**: Minimum 4GB RAM recommended
- **Python**: Version 3.8+ required

## 🤝 Contributing

Feel free to enhance the system by:
- Adding new keyword extraction methods
- Implementing additional similarity metrics
- Improving the web interface
- Adding database integration

## 📄 License

This project uses various open-source libraries. See individual module licenses.

## 🆘 Troubleshooting

### Issue: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Issue: KeyBERT slow on CPU
Use TF-IDF method instead or install CUDA support

### Issue: Import errors
Make sure you're in the correct directory and virtual environment is activated

## 📧 Support

For issues or questions, please refer to the documentation in each module file.

---

**Built with**: spaCy, Gensim, scikit-learn, YAKE, KeyBERT, Sentence-Transformers   


# System Architecture

## Overall Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                           │
│  ┌────────────┐              ┌──────────────┐               │
│  │  PDF File  │              │  Text Input  │               │
│  └─────┬──────┘              └──────┬───────┘               │
└────────┼─────────────────────────────┼──────────────────────┘
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXTRACTION LAYER                           │
│               (pdf_extractor.py)                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  • PyPDF2 / pdfplumber                           │       │
│  │  • Text extraction                                │       │
│  │  • Metadata extraction                            │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                          │
│               (preprocessing.py)                             │
│  ┌──────────────────────────────────────────────────┐       │
│  │  • Text cleaning (spaCy)                         │       │
│  │  • Lemmatization                                  │       │
│  │  • Stopword removal                               │       │
│  │  • Noun phrase extraction                         │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│  SUMMARIZATION   │          │    KEYWORD       │
│     LAYER        │          │   EXTRACTION     │
│ (summarization   │          │     LAYER        │
│      .py)        │          │ (keyword_        │
│                  │          │  extraction.py)  │
│  ┌────────────┐  │          │  ┌────────────┐  │
│  │  TextRank  │  │          │  │   YAKE     │  │
│  │   (sumy)   │  │          │  │   RAKE     │  │
│  └────────────┘  │          │  │  KeyBERT   │  │
│                  │          │  └────────────┘  │
└─────────┬────────┘          └────────┬─────────┘
          │                            │
          └───────────┬────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              SIMILARITY & RANKING LAYER                      │
│            (similarity_ranking.py)                           │
│  ┌──────────────────────────────────────────────────┐       │
│  │                                                    │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │       │
│  │  │  TF-IDF  │  │   BERT   │  │  Hybrid  │       │       │
│  │  │          │  │          │  │          │       │       │
│  │  │ (scikit) │  │(Sentence │  │(Combined)│       │       │
│  │  │          │  │Transform)│  │          │       │       │
│  │  └──────────┘  └──────────┘  └──────────┘       │       │
│  │                                                    │       │
│  │  • Cosine Similarity Computation                  │       │
│  │  • Citation Ranking                               │       │
│  │  • Novelty Score Calculation                      │       │
│  └──────────────────────────────────────────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  • Summary                                        │       │
│  │  • Keywords                                       │       │
│  │  • Ranked Prior Art Citations                    │       │
│  │  • Novelty Score & Assessment                    │       │
│  │  • JSON Report                                    │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Module Interaction Diagram

```
                    ┌─────────────────┐
                    │   pipeline.py   │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────┐  ┌──────────────────┐
│ pdf_extractor   │  │preprocessing│  │ summarization    │
│      .py        │  │    .py      │  │      .py         │
└─────────────────┘  └─────────────┘  └──────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────┐  ┌──────────────────┐
│   keyword_      │  │ similarity_ │  │  User Interface  │
│  extraction.py  │  │ ranking.py  │  │ (web/notebook)   │
└─────────────────┘  └─────────────┘  └──────────────────┘
```

## Data Flow

```
1. INPUT
   ├─→ Raw Text / PDF File
   └─→ Prior Art Documents

2. EXTRACTION & PREPROCESSING
   ├─→ Extract text from PDF
   ├─→ Clean and normalize
   └─→ Remove noise

3. ANALYSIS
   ├─→ Generate summary (TextRank)
   ├─→ Extract keywords (YAKE/RAKE/KeyBERT)
   └─→ Prepare for comparison

4. COMPARISON
   ├─→ Convert to vectors (TF-IDF/BERT)
   ├─→ Compute similarity scores
   └─→ Rank prior art by relevance

5. ASSESSMENT
   ├─→ Calculate novelty score
   ├─→ Identify most similar prior art
   └─→ Generate recommendations

6. OUTPUT
   ├─→ Formatted report
   ├─→ JSON file with all results
   └─→ Visualization (optional)
```

## Technology Stack

```
┌────────────────────────────────────────┐
│         Python 3.8+                    │
└────────────────────────────────────────┘
              │
    ┌─────────┴──────────┐
    │                    │
┌───▼──────────┐  ┌──────▼────────┐
│   NLP Core   │  │  ML/Similarity│
├──────────────┤  ├───────────────┤
│  • spaCy     │  │ • scikit-learn│
│  • NLTK      │  │ • gensim      │
│  • Gensim    │  │ • sentence-   │
│              │  │   transformers│
└──────────────┘  └───────────────┘
    │                    │
    │  ┌────────────────┐│
    │  │  Keyword       ││
    │  │  Extraction    ││
    │  ├────────────────┤│
    │  │ • YAKE         ││
    │  │ • RAKE         ││
    │  │ • KeyBERT      ││
    │  └────────────────┘│
    │                    │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Data Processing   │
    ├────────────────────┤
    │  • pandas          │
    │  • numpy           │
    └────────────────────┘
              │
    ┌─────────▼──────────┐
    │  PDF Processing    │
    ├────────────────────┤
    │  • PyPDF2          │
    │  • pdfplumber      │
    └────────────────────┘
              │
    ┌─────────▼──────────┐
    │  Web Interface     │
    ├────────────────────┤
    │  • Flask           │
    │  • HTML/CSS/JS     │
    └────────────────────┘
```

## Algorithm Selection Guide

```
┌───────────────────────────────────────────────────────┐
│             KEYWORD EXTRACTION                        │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Fast & Simple     →  RAKE                           │
│  Best Balance      →  YAKE  ⭐ RECOMMENDED           │
│  Most Accurate     →  KeyBERT                        │
│  All Combined      →  extract_combined()             │
│                                                       │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│           SIMILARITY COMPUTATION                      │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Fast              →  TF-IDF                         │
│  Most Accurate     →  BERT                           │
│  Best Overall      →  Hybrid  ⭐ RECOMMENDED         │
│                                                       │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│              SUMMARIZATION                            │
├───────────────────────────────────────────────────────┤
│                                                       │
│  For Long Text     →  TextRank (sumy)                │
│  For Short Text    →  Simple Extraction              │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## Performance Characteristics

| Component | Speed | Accuracy | Resource Use |
|-----------|-------|----------|--------------|
| RAKE | ⚡⚡⚡ | ⭐⭐ | 💾 Low |
| YAKE | ⚡⚡ | ⭐⭐⭐ | 💾 Low |
| KeyBERT | ⚡ | ⭐⭐⭐⭐ | 💾💾 Medium |
| TF-IDF | ⚡⚡⚡ | ⭐⭐⭐ | 💾 Low |
| BERT | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾 High |
| Hybrid | ⚡⚡ | ⭐⭐⭐⭐ | 💾💾 Medium |

Legend:
- ⚡ = Speed (more = faster)
- ⭐ = Accuracy (more = better)
- 💾 = Resource usage (more = higher)