# SMARTNLP

SMARTNLP is an AI-powered chatbot built using transformer models (like GPT) for contextual text generation. It also features text summarization, sentiment analysis, and named entity recognition (NER) functionalities.

## Features
- **AI Chatbot**: Chat with a conversational bot powered by GPT.
- **Text Summarization**: Summarize long paragraphs into concise summaries.
- **Sentiment Analysis**: Analyze the sentiment of a text (positive, negative, or neutral).
- **Named Entity Recognition**: Identify and classify named entities in text (e.g., names, locations, dates).

## Requirements
- Python 3.8 or higher
- MongoDB for storing conversation history

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/nikhilbhateja22/SMARTNLP.git
   cd SMARTNLP
Install dependencies:

pip install -r requirements.txt
Start MongoDB service:

sudo service mongod start
Run the Flask app:

python app.py
Access the application at http://127.0.0.1:5000.
