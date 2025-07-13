import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer

def create_embeddings_and_index(csv_path, texts_path, index_path):
    # Load data
    df = pd.read_csv(csv_path)

    important_cols = ["Loan_Status", "Married", "Credit_History", "LoanAmount"]

    texts = []
   
    for _, row in df.iterrows():
       important_cols = ["Loan_Status", "Married", "Credit_History", "LoanAmount"]
       if any(pd.isna(row[col]) for col in important_cols):
           continue  # skip incomplete rows
       row = row.fillna("Missing")
       text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
       texts.append(text)


    # Save texts for later retrieval
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)

    # Load embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, convert_to_numpy=True).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index to disk
    faiss.write_index(index, index_path)

    print(f"Successfully embedded {len(texts)} chunks and saved index.")

if __name__ == "__main__":
    create_embeddings_and_index(
        csv_path="dataset/Training Dataset.csv",
        texts_path="embeddings/texts.pkl",
        index_path="embeddings/index.faiss"
    )

