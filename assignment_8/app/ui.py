# app/ui.py

import streamlit as st
import pickle
import faiss
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot import get_answer

with open("embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)

index = faiss.read_index("embeddings/index.faiss")

st.title("Loan Approval Chatbot (Gemini + RAG)")
st.write("Ask anything about the loan dataset...")

query = st.text_input("Your Question")

if query:
    response = get_answer(query, texts, index)
    st.markdown("### Gemini Answer:")
    st.write(response)



