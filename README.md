## NLP Article Summarisation  
This is a simple solution that uses pretrained NLP model(google/pegasus-cnn) to summarise input articles and streamlit for web UI.

## Demo
[View Demo](http://bluelantern.tplinkdns.com:8501/)

### Features
- Dealing with long inputs (truncate)                Done  
- Chunking long inputs for better performance        In Progress  
- Customise min and max length for input and output  Done  
- Multi language support                             In Progress  
- Final Tuning                                       Planned  

### Installation
0. Using venv is recommended.  
0.1 python -m venv venv (first time inni only)  
0.2 venv\Scripts\activate  

1. install requirements.txt
2. Optional: CUDA if you want to use GPU (faster)
3. streamlit run app.py
4. access on localhost:8501 (default port is 8501)