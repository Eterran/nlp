## Multilingual News Article Summarizer

A complete multilingual news summarization solution using Google Pegasus and Facebook NLLB-200 models with Streamlit web interface and comprehensive evaluation pipeline.

## Demo

[View Demo](http://bluelantern.tplinkdns.com:8501/)

### Features

- ✅ **Multilingual Support**: Auto-detects 70+ languages, translates to English for summarization, then back-translates
- ✅ **Chunking for Long Articles**: Handles unlimited text length with overlapping token-based chunking
- ✅ **Customizable Parameters**: Adjustable min/max summary lengths and chunk processing
- ✅ **Evaluation Pipeline**: ROUGE metrics on CNN/DailyMail (English) and MLSUM (French) datasets
- ✅ **Interactive Dashboard**: Compare results with official Pegasus benchmarks
- ✅ **Navigation System**: Seamless switching between summarizer and evaluation views

### Models Used

- **Summarization**: [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
- **Translation**: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)

### Installation

1. **Virtual Environment** (recommended):

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**:

   ```powershell
   pip install -r requirements.txt
   ```

3. **Optional**: Install CUDA for GPU acceleration (faster processing)

4. **Run Application**:

   ```powershell
   streamlit run app.py
   ```

5. **Access**: Open browser to `http://localhost:8501`

### File Structure

```
📁 nlp/
├── 📄 app.py                          # Main Streamlit application with navigation
├── 📄 summariser.py                   # Core multilingual summarizer class
├── 📄 evaluation.py                   # Evaluation pipeline for ROUGE metrics
├── 📄 evaluation_dashboard.py         # Results dashboard with benchmark comparisons
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This documentation
├── 📊 evaluation_results_en.csv      # English evaluation results
├── 📊 evaluation_results_fr.csv      # French evaluation results
└── 📊 evaluation_results_summary.csv # Aggregated metrics summary
```

### Quick Start

```powershell
# Clone and setup
git clone <repository-url>
cd nlp

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Usage

1. **News Summarizer**: Paste any article (70+ languages supported) and get intelligent summaries
2. **Evaluation Dashboard**: View ROUGE metrics and compare with official Pegasus benchmarks
3. **Navigation**: Use sidebar to switch between summarizer and evaluation views

### Technical Details

- **Chunking**: 462-token chunks with 50-token overlap for long articles
- **Languages**: Auto-detection with langdetect, NLLB-200 translation support
- **Evaluation**: 25 samples per language, ROUGE-1/2/L metrics
- **GPU Support**: Automatic CUDA detection for faster processing

### Team

Made by **Group 31**: Loh Lit Hoong, John Ong Ming Hom, Liew Jin Sze, Kueh Pang Lang
