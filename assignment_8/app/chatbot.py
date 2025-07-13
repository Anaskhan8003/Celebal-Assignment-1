import os
import pickle
import faiss
import pandas as pd
import numpy as np
import google.generativeai as genai
from utils.retrieve import retrieve_similar_chunks

# Load the dataset (if needed elsewhere)
df = pd.read_csv("dataset/Training Dataset.csv")

# --- Use absolute paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "index.faiss")
TEXTS_PATH = os.path.join(BASE_DIR, "embeddings", "texts.pkl")

# Load texts and FAISS index
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

faiss_index = faiss.read_index(INDEX_PATH)

# Configure Gemini
genai.configure(api_key="AIzaSyDX7rx3i1QdfRte5QioA5XAF7biqeOUJE4")  # Replace with your real API key if not using env var

# Use gemini-1.5-flash (recommended free-tier model)
model = genai.GenerativeModel("gemini-1.5-flash")

def get_answer(query, texts, index):
    context = retrieve_similar_chunks(query, texts, index, k=9)

    prompt = f"""
You are a highly capable data analyst AI assistant.

You will be given a dataset and a question. Your job is to:
1. Answer numeric questions using exact calculations (e.g., averages, counts).
2. Answer general/comparative questions by analyzing relevant trends (e.g., compare graduates vs. non-graduates).
3. Provide a confident, concise conclusion. Do not say the dataset is too small unless it has fewer than 5 relevant entries.
4. Exclude missing data from calculations and mention if any data was skipped.
5. Avoid mentioning missing data unless it affects the numeric result significantly.

**Dataset:**
{context}

**Question:**
{query}

**Answer:**
"""




    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting answer: {e}"

