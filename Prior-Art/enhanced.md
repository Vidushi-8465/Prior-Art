# Enhanced Patent Analysis System - User Guide

## 🎯 What's New

Your system now:
- ✅ Takes **PDF files** as input (instead of text)
- ✅ Extracts **metadata** (IPC codes, CPC codes, publication year, citations)
- ✅ Expands **keywords with full forms** (e.g., NLP → Natural Language Processing)
- ✅ Outputs results in **CSV format** ready for Excel

## 📋 Output CSV Format

```
title | abstract | ipc_codes | cpc_codes | keywords | citations | publication_year
```

### Example Output:
```csv
title,abstract,ipc_codes,cpc_codes,keywords,citations,publication_year
"Semantic Patent Search","A system for...",G06F 40/30,G06F 40/30,"nlp (Natural Language Processing); machine learning; patent search",US 1234567,2024
```

---

## 🚀 Quick Start

### Option 1: Interactive Mode (Easiest)

```bash
cd src
python enhanced_demo.py
```

Then follow the prompts:
1. Choose option 1 (Interactive mode)
2. Enter path to your PDF file
3. Enter number of keywords (or press Enter for default 15)
4. Wait for analysis
5. Get your CSV file!

### Option 2: Python Script

```python
from enhanced_pipeline import PatentAnalysisPipeline

# Initialize
pipeline = PatentAnalysisPipeline(output_dir="../data/output")

# Analyze single PDF
csv_path = pipeline.run_pipeline("path/to/your/patent.pdf")

# Or analyze multiple PDFs
csv_path = pipeline.run_pipeline([
    "patent1.pdf",
    "patent2.pdf",
    "patent3.pdf"
])

print(f"CSV saved to: {csv_path}")
```

---

## 📂 File Structure

```
prior_art_search/
├── src/
│   ├── patent_pdf_extractor.py      # Extract PDF + metadata
│   ├── keyword_expander.py          # Expand keywords (NLP → Natural Language Processing)
│   ├── csv_output_generator.py      # Generate CSV output
│   ├── enhanced_pipeline.py         # Main pipeline
│   └── enhanced_demo.py             # Demo script
│
└── data/
    ├── input/                        # Put your PDF files here
    └── output/                       # CSV files will be saved here
```

---

## 🔍 What Gets Extracted

### From PDF Metadata & Text:

1. **Title** - Patent/document title
2. **Abstract** - Patent abstract or generated summary
3. **IPC Codes** - International Patent Classification codes
   - Example: `G06F 40/30`, `H04L 29/06`
4. **CPC Codes** - Cooperative Patent Classification codes
   - Example: `G06F 40/30`
5. **Keywords** - Extracted and expanded with full forms
   - Example: `nlp (Natural Language Processing); ml (Machine Learning); ai (Artificial Intelligence)`
6. **Citations** - Referenced patents
   - Example: `US 1234567, US 7654321, EP 0123456`
7. **Publication Year** - Year of publication
   - Example: `2024`

---

## 💡 Usage Examples

### Example 1: Single Patent Analysis

```python
from enhanced_pipeline import PatentAnalysisPipeline

pipeline = PatentAnalysisPipeline()

# Analyze one patent
csv_file = pipeline.run_pipeline(
    pdf_paths="my_patent.pdf",
    num_keywords=20,  # Extract 20 keywords
    output_filename="my_analysis.csv"
)
```

### Example 2: Batch Analysis

```python
import glob

pipeline = PatentAnalysisPipeline()

# Get all PDFs in a folder
pdf_files = glob.glob("../data/input/*.pdf")

# Analyze all at once
csv_file = pipeline.run_pipeline(
    pdf_paths=pdf_files,
    num_keywords=15,
    output_filename="batch_analysis.csv"
)
```

### Example 3: Custom Analysis

```python
pipeline = PatentAnalysisPipeline()

# Analyze single PDF and view results before saving
result = pipeline.analyze_single_pdf("patent.pdf", num_keywords=15)

# Preview the results
pipeline.print_result_preview(result)

# Save to CSV
csv_file = pipeline.generate_output_csv([result], "custom_output.csv")
```

---

## 🔧 Keyword Expansion

The system automatically expands acronyms and abbreviations:

| Original | Expanded |
|----------|----------|
| nlp | Natural Language Processing |
| ml | Machine Learning |
| ai | Artificial Intelligence |
| cnn | Convolutional Neural Network |
| api | Application Programming Interface |
| iot | Internet of Things |
| gps | Global Positioning System |

**200+ acronyms** are built into the system!

### Auto-Detection

The system also detects acronyms **from the document itself**:

If your PDF contains:
> "Natural Language Processing (NLP) is used..."

The system will detect this and expand `nlp` to `Natural Language Processing` automatically.

---

## 📊 CSV Output Details

### Standard CSV (Default)
Columns: `title, abstract, ipc_codes, cpc_codes, keywords, citations, publication_year`

### Detailed CSV (Optional)
Additional columns: `filename, keywords_expanded, num_pages, summary`

To generate detailed CSV:
```python
csv_path = pipeline.csv_generator.generate_detailed_csv(
    results,
    output_path="detailed_analysis.csv"
)
```

---

## 🎨 Customization

### Change Number of Keywords

```python
# Extract more keywords
csv_file = pipeline.run_pipeline(
    pdf_paths="patent.pdf",
    num_keywords=30  # Default is 15
)
```

### Add Custom Acronyms

Edit `keyword_expander.py` and add to the `ACRONYM_DATABASE`:

```python
ACRONYM_DATABASE = {
    'your_acronym': 'Your Full Form',
    'xyz': 'Example Full Form',
    # ... existing acronyms
}
```

### Change Output Directory

```python
pipeline = PatentAnalysisPipeline(output_dir="path/to/your/output")
```

---

## 🐛 Troubleshooting

### PDF Not Extracting Text

**Problem:** Empty or garbled text extracted.

**Solutions:**
1. PDF might be image-based (scanned) - needs OCR
2. Try different extraction method in code
3. Check if PDF is password protected

### Missing IPC/CPC Codes

**Problem:** Codes not found in output.

**Reasons:**
- Document is not a patent
- Codes are in non-standard format
- PDF quality is poor

**Solution:** Manually add codes if you know them.

### Keywords Not Expanding

**Problem:** Acronyms not being expanded.

**Solutions:**
1. Add your specific acronyms to `ACRONYM_DATABASE`
2. Make sure acronym is defined in the document (e.g., "Natural Language Processing (NLP)")

---

## 📝 Step-by-Step Workflow

### Complete Analysis Workflow:

1. **Prepare PDFs**
   - Place PDF files in `data/input/` folder
   - Or note their file paths

2. **Run Analysis**
   ```bash
   cd src
   python enhanced_demo.py
   ```

3. **Choose Mode**
   - Interactive: Enter PDF path when prompted
   - Or write a Python script

4. **Get Results**
   - CSV file saved to `data/output/`
   - Contains all extracted information

5. **Open in Excel**
   - Double-click CSV file
   - Or import into Excel/Google Sheets

---

## 💻 Command Examples

### Quick Analysis

```bash
# Interactive mode
python enhanced_demo.py

# Direct Python
python -c "from enhanced_pipeline import PatentAnalysisPipeline; PatentAnalysisPipeline().run_pipeline('patent.pdf')"
```

### Batch Processing Script

Create `batch_analyze.py`:
```python
from enhanced_pipeline import PatentAnalysisPipeline
import glob

pipeline = PatentAnalysisPipeline()

# Get all PDFs
pdfs = glob.glob("../data/input/*.pdf")

# Analyze
csv = pipeline.run_pipeline(pdfs, num_keywords=20)

print(f"Done! CSV: {csv}")
```

Run:
```bash
python batch_analyze.py
```

---

## 📈 Performance

- **Single PDF**: ~10-30 seconds
- **Batch (10 PDFs)**: ~2-5 minutes
- **Speed depends on**:
  - PDF size
  - Number of pages
  - Number of keywords requested

---

## 🎯 Best Practices

1. **PDF Quality**: Use text-based PDFs (not scanned images)
2. **Keywords**: Start with 15, increase if needed
3. **Batch Processing**: Process multiple files at once for efficiency
4. **Review Output**: Check CSV for accuracy, especially IPC/CPC codes
5. **Backup**: Keep original PDFs

---

## 🔗 Integration

### With Excel
- Open CSV directly in Excel
- Or import as external data
- Use Excel functions to analyze further

### With Google Sheets
- Import CSV via File → Import
- Share with collaborators
- Use Google Sheets formulas

### With Database
```python
import pandas as pd

# Read CSV
df = pd.read_csv("patent_analysis.csv")

# Convert to database
df.to_sql('patents', con=database_connection)
```

---

## 📞 Getting Help

If something doesn't work:

1. Check file path is correct
2. Verify PDF is not corrupted
3. Look at error message
4. Try with a different PDF first

---

## ✅ Checklist

Before running:
- [ ] PDFs are ready
- [ ] Output directory exists
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] spaCy model downloaded (`python -m spacy download en_core_web_sm`)
- [ ] NLTK data downloaded

After running:
- [ ] CSV file generated
- [ ] Check output for accuracy
- [ ] Open in Excel to verify format

---

**You're ready to go!** 🚀