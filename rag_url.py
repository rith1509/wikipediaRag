import base64
from io import BytesIO
from urllib.parse import urljoin, urlparse
import requests
from multiprocessing import freeze_support

from bs4 import BeautifulSoup
from PIL import Image
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

# --- 1. CONFIGURATION ---
SCRAPINGBEE_API_KEY = "6PW5WU0SBXOINFGYCPRIQRU1UKMT8THGDHVXB278OHGSJJS4APG63CGN0Y98IR4U0CEK597YX42EUU6K" 
WEBSITE_URL = "https://www.eneba.com/top-up-games?af_id=operagenshin&utm_medium=af&utm_source=operagenshin" 
TEXT_EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3:8b-instruct-q4_K_M"
IMAGE_CAPTION_MODEL = "bakllava"

# --- 2. IMAGE CAPTIONING FUNCTION ---
def get_image_caption(image_bytes: bytes, model: str) -> str:
    llm_with_image = ChatOllama(model=model, temperature=0)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    msg = llm_with_image.invoke(
        [HumanMessage(content=[
            {"type": "text", "text": "Describe this image in detail, focusing on people, actions, and artistic style."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_image}"},
        ])]
    )
    return msg.content

# --- MAIN LOGIC ---
def main():
    if not SCRAPINGBEE_API_KEY or SCRAPINGBEE_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please paste your ScrapingBee API key into the script.")
        return

    print(f"Step 1: Scraping '{WEBSITE_URL}'...")
    try:
        response = requests.get(
            url='https://app.scrapingbee.com/api/v1/',
            params={'api_key': SCRAPINGBEE_API_KEY, 'url': WEBSITE_URL, 'render_js': 'true', 'premium_proxy': 'true'},
            timeout=180
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        print("Successfully retrieved and parsed HTML.")
    except requests.exceptions.RequestException as e:
        print(f"Scraping failed: {e}")
        return

    print("-" * 30)

    # --- Text Extraction Pipeline ---
    print("Step 2: Processing text content...")
    # Make a copy of the soup for text extraction before modifying it
    text_soup = BeautifulSoup(str(soup), "html.parser")
    for tag in text_soup(['script', 'style', 'nav', 'footer', 'header', 'table', 'aside', 'figure']):
        tag.decompose()
    text_content = text_soup.get_text(separator='\n', strip=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splits = text_splitter.create_documents([text_content])
    text_embeddings = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL)
    text_vectorstore = FAISS.from_documents(text_splits, text_embeddings)
    text_retriever = text_vectorstore.as_retriever(k=3)
    print("Text content processed.")
    print("-" * 30)

    # --- Image Extraction and Enrichment Pipeline ---
    print("Step 3: Processing images...")
    enriched_image_docs = []
    # Use a more precise selector for images within figures, which often have captions
    figures = soup.find_all('figure')
    for figure in figures:
        try:
            img_tag = figure.find('img')
            figcaption = figure.find('figcaption')
            
            # Use the figcaption as the primary text context
            original_caption = figcaption.get_text(strip=True) if figcaption else "No caption found"
            
            src = img_tag.get("src") if img_tag else None
            if not src or not isinstance(src, str): continue
            if src.startswith('//'): src = 'https:' + src
            if not src.startswith('http'): continue
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            image_response = requests.get(src, timeout=15, headers=headers)
            image_response.raise_for_status()
            image_bytes = BytesIO(image_response.content).read()
            
            if len(image_bytes) > 8192:
                ai_caption = get_image_caption(image_bytes, IMAGE_CAPTION_MODEL)
                
                # Create a structured document that prioritizes the original caption
                enriched_doc = f"Image with original caption: '{original_caption}'.\nAI-generated visual description: {ai_caption}"
                print(f"Generated enriched doc for image with caption: '{original_caption[:50]}...'")
                enriched_image_docs.append(enriched_doc)
        except Exception:
            continue

    if enriched_image_docs:
        image_vectorstore = FAISS.from_texts(enriched_image_docs, text_embeddings)
        image_retriever = image_vectorstore.as_retriever(k=2)
        print(f"Successfully processed {len(enriched_image_docs)} images.")
    else:
        def dummy_retriever(query): return ["No valid image context found."]
        image_retriever = dummy_retriever
        print("No valid images were processed.")

    print("-" * 30)
    
    # --- Unified RAG Chain using MergerRetriever ---
    print("Step 4: Setting up and running the unified RAG chain...")
    llm = ChatOllama(model=LLM_MODEL)
    
    from langchain.retrievers import MergerRetriever
    lotr = MergerRetriever(retrievers=[text_retriever, image_retriever])
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert research assistant. Answer the user's question based ONLY on the following context retrieved from a webpage. The context contains both text from the article and detailed descriptions of images.

        **CONTEXT:**
        {context}

        **QUESTION:** {question}

        **ANSWER:**"""
    )

    rag_chain = (
        {"context": lotr, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # --- Query the RAG ---
    query = "Describe stumble guys image"
    print(f"User Query: {query}\n")
    print("=" * 50)
    print("Streaming Final Answer:")
    print("=" * 50)
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
    print("\n")
    print("=" * 50)


if __name__ == '__main__':
    freeze_support()
    main()