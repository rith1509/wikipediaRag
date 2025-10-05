# 🕸️ Webpage RAG Scraper & Image Captioning Assistant

A multi-modal Retrieval-Augmented Generation (RAG) pipeline that scrapes a webpage, processes its text and image content, and generates AI-enhanced summaries using LangChain, Ollama, and ScrapingBee.

This tool extracts and embeds webpage data (text and images) into FAISS vector stores, enabling natural language queries that consider both written content and visual elements.

## 📚 Table of Contents

Overview

Features

Architecture

Installation

Configuration

Usage

Example Output

Dependencies

Troubleshooting

Contributors

License

## 🧠 Overview

This project demonstrates an end-to-end web intelligence pipeline:

Scrape a live website using ScrapingBee
.

Extract and embed text using FAISS and OllamaEmbeddings.

Caption and embed images using Bakllava (multi-modal LLM).

Merge retrievers for both modalities using LangChain’s MergerRetriever.

Query the combined dataset using a RAG chain with a local Llama 3 model.

## 🚀 Features

✅ Web scraping with JavaScript rendering
✅ Text chunking & semantic embedding
✅ Automatic image caption generation with AI
✅ Multi-modal retrieval (text + image descriptions)
✅ FAISS vector storage for efficient similarity search
✅ RAG-based natural language Q&A over scraped content

## 🏗️ Architecture
 ┌───────────────────┐
 │  Target Website   │
 └────────┬──────────┘
          │
   (1) ScrapingBee API
          │
 ┌────────▼──────────┐
 │ BeautifulSoup HTML│
 └────────┬──────────┘
          │
   ┌──────▼──────┐     ┌──────────────────────┐
   │ Text Extract │     │  Image Extract + AI  │
   └──────┬──────┘     └──────────┬───────────┘
          │                       │
   (2) Recursive Splitter   (3) Bakllava Caption
          │                       │
   ┌──────▼──────────┐       ┌────▼──────────┐
   │ Text Embeddings │       │ Image Embeds  │
   │ (FAISS store)   │       │ (FAISS store) │
   └──────┬──────────┘       └────┬──────────┘
          │                       │
          └──────┬────────────────┘
                 ▼
         MergerRetriever
                 │
          ChatOllama (Llama 3)
                 │
              (Output)

## ⚙️ Installation
### Clone this repository
git clone https://github.com/yourusername/webpage-rag-scraper.git
cd webpage-rag-scraper

### Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

### Install dependencies
pip install -r requirements.txt

## 🔧 Configuration

Edit the following constants at the top of the script:

SCRAPINGBEE_API_KEY = "YOUR_SCRAPINGBEE_API_KEY"
WEBSITE_URL = "https://www.example.com"
TEXT_EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:8b-instruct-q4_K_M"
IMAGE_CAPTION_MODEL = "bakllava"


⚠️ Never commit your API keys to public repositories.

You’ll also need:

Ollama installed and running locally

The specified models pulled via ollama pull llama3:8b-instruct-q4_K_M and ollama pull bakllava

## ▶️ Usage

Run the main script:

python main.py


It performs:

Webpage scraping

Text + image content extraction

Embedding and vector storage

Query execution via RAG

You can modify the query near the end of the script:

query = "Describe stumble guys image"

## 💬 Example Output
Step 1: Scraping 'https://www.eneba.com/top-up-games'...
Successfully retrieved and parsed HTML.
Step 2: Processing text content...
Text content processed.
Step 3: Processing images...
Generated enriched doc for image with caption: 'Top-up Stumble Guys Gems...'
Successfully processed 4 images.
Step 4: Running unified RAG chain...

User Query: Describe stumble guys image
--------------------------------------------------
Streaming Final Answer:
The image shows colorful cartoon characters jumping and racing in obstacle courses, typical of “Stumble Guys.” The style is playful and 3D-rendered...
--------------------------------------------------

## 📦 Dependencies

requests

beautifulsoup4

Pillow

pydantic

langchain

langchain-community

langchain-core

langchain-ollama

faiss-cpu

multiprocessing

Install all dependencies with:

pip install -r requirements.txt

## 🧩 Troubleshooting
Problem	Possible Fix
❌ “Please paste your ScrapingBee API key”	Add a valid API key in the config section.
⚠️ Ollama not connecting	Make sure ollama serve is running locally.
🖼️ No images processed	Check if the site uses lazy-loading (try enabling render_js=True).
🧠 Poor AI captions	Ensure you’re using bakllava or another vision-capable model.
👥 Contributors

Author: [Your Name]
Maintainer: [Your GitHub Handle]
Contributions, pull requests, and suggestions are welcome!

📜 License

This project is licensed under the MIT License — feel free to modify and distribute with attribution.
