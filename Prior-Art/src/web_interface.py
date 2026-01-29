"""
Web Interface for Prior Art Search
-----------------------------------
Simple Flask web application for easy interaction with the pipeline.
"""

from flask import Flask, request, render_template_string, jsonify
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pipeline import PriorArtPipeline

app = Flask(__name__)
pipeline = PriorArtPipeline(output_dir="../data/output")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Prior Art Search System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #34495e;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #3498db;
        }
        button {
            background: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #2980b9;
        }
        button:disabled {
            background: #95a5a6;
            cursor: not-allowed;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #ecf0f1;
            border-radius: 5px;
            display: none;
        }
        .results.show {
            display: block;
        }
        .section {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 5px;
        }
        .section h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .keyword {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 12px;
            margin: 3px;
            border-radius: 15px;
            font-size: 13px;
        }
        .citation {
            padding: 10px;
            margin: 10px 0;
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 3px;
        }
        .score {
            font-weight: bold;
            color: #e74c3c;
        }
        .novelty-high { color: #27ae60; }
        .novelty-medium { color: #f39c12; }
        .novelty-low { color: #e74c3c; }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Prior Art Search System</h1>
        <p>Enter your invention description to analyze novelty and find similar prior art.</p>
        
        <form id="searchForm">
            <div class="form-group">
                <label for="invention">Invention Description (2-3 paragraphs):</label>
                <textarea id="invention" rows="8" placeholder="Describe your invention in detail..."></textarea>
            </div>
            
            <div class="form-group">
                <label for="priorArt">Prior Art Documents (one per line):</label>
                <textarea id="priorArt" rows="6" placeholder="Enter prior art documents, separated by line breaks...
Example:
A system for image recognition using neural networks.
Methods for natural language processing in search engines.
..."></textarea>
            </div>
            
            <button type="submit">Analyze</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing... This may take a moment.</p>
        </div>
        
        <div class="results" id="results"></div>
    </div>
    
    <script>
        document.getElementById('searchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const invention = document.getElementById('invention').value;
            const priorArt = document.getElementById('priorArt').value;
            
            if (!invention || !priorArt) {
                alert('Please fill in both fields!');
                return;
            }
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');
            
            // Prepare data
            const priorArtDocs = priorArt.split('\\n')
                .filter(line => line.trim())
                .map((text, idx) => ({
                    text: text.trim(),
                    metadata: { id: `PA${idx + 1}` }
                }));
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        invention: invention,
                        prior_art: priorArtDocs
                    })
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        });
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            const analysis = data.analysis;
            const comparison = data.prior_art_comparison;
            const novelty = comparison.novelty_metrics;
            
            let noveltyClass = 'novelty-low';
            let noveltyText = 'LOW';
            if (novelty.novelty_score > 0.7) {
                noveltyClass = 'novelty-high';
                noveltyText = 'HIGH';
            } else if (novelty.novelty_score > 0.4) {
                noveltyClass = 'novelty-medium';
                noveltyText = 'MODERATE';
            }
            
            let html = `
                <div class="section">
                    <h3>📝 Summary</h3>
                    <p>${analysis.summary || 'No summary available'}</p>
                </div>
                
                <div class="section">
                    <h3>🔑 Keywords</h3>
                    ${analysis.keywords.unique_keywords.map(kw => 
                        `<span class="keyword">${kw}</span>`
                    ).join('')}
                </div>
                
                <div class="section">
                    <h3>💡 Novelty Assessment</h3>
                    <p><strong>Novelty Score:</strong> 
                        <span class="score ${noveltyClass}">${(novelty.novelty_score * 100).toFixed(1)}%</span>
                        (${noveltyText})
                    </p>
                    <p><strong>Max Similarity to Prior Art:</strong> ${(novelty.max_similarity * 100).toFixed(1)}%</p>
                    <p><strong>Average Similarity:</strong> ${(novelty.avg_similarity * 100).toFixed(1)}%</p>
                </div>
                
                <div class="section">
                    <h3>📚 Top 5 Similar Prior Art</h3>
            `;
            
            comparison.ranked_citations.slice(0, 5).forEach(([rank, doc, score]) => {
                html += `
                    <div class="citation">
                        <strong>Rank ${rank}</strong> | 
                        <span class="score">Similarity: ${(score * 100).toFixed(1)}%</span>
                        <p>${doc.text.substring(0, 200)}${doc.text.length > 200 ? '...' : ''}</p>
                    </div>
                `;
            });
            
            html += '</div>';
            
            resultsDiv.innerHTML = html;
            resultsDiv.classList.add('show');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    API endpoint for analyzing invention and prior art.
    
    Expected JSON:
    {
        "invention": "invention description text",
        "prior_art": [
            {"text": "prior art 1", "metadata": {...}},
            {"text": "prior art 2", "metadata": {...}}
        ]
    }
    """
    try:
        data = request.get_json()
        invention = data.get('invention', '')
        prior_art = data.get('prior_art', [])
        
        if not invention or not prior_art:
            return jsonify({'error': 'Missing invention or prior art data'}), 400
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            invention_input=invention,
            prior_art_docs=prior_art,
            is_file=False,
            similarity_method="tfidf",  # Faster for demo
            save_results=True
        )
        
        # Convert to JSON-serializable format
        response = {
            'analysis': results['analysis'],
            'prior_art_comparison': {
                'ranked_citations': [
                    (rank, doc, float(score)) 
                    for rank, doc, score in results['prior_art_comparison']['ranked_citations']
                ],
                'novelty_metrics': results['prior_art_comparison']['novelty_metrics'],
                'similarity_method': results['prior_art_comparison']['similarity_method']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Starting Prior Art Search Web Interface")
    print("="*70)
    print("\nOpen your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)