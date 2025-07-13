import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("AIzaSyAtF2HVMJwhatciH0vxpJciQjFvPJHcOTk"))
model = genai.GenerativeModel("gemini-pro")

def generate_answer(context, question):
    prompt = f"""You are a helpful assistant answering questions about loan applications.

Context:
{context}

Question:
{question}

Answer:"""
    response = model.generate_content(prompt)
    return response.text


