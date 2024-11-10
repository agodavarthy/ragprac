from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import chromadb
import ollama

#Initializing models
LLM_MODEL_NAME= "llama2"
model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model, adjust as needed
client = chromadb.Client()

def read_dataset():
    dataset = load_dataset("neuralwork/arxiver", streaming=True)
    train_df = pd.DataFrame(dataset['train'])
    corpus = train_df['title'] + " " + train_df['abstract']
    corpus = corpus[:10]
    return corpus

def addDocsChromaColl(corpus, embeddings):
    for i, (doc, embedding) in enumerate(zip(corpus, embeddings)):
        collection.add(
            ids=str(i),
            documents=[doc],
            metadatas=[{"id": i}],
            embeddings=[embedding.tolist()]
        )

def queryChroma(query):
    # Convert the query into an embedding
    query_embedding = model.encode([query])[0]

    # Perform the similarity search in the corpus
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5  # Number of similar results you want
    )
    return results

corpus = read_dataset()

# Generate embeddings for the corpus
embeddings = model.encode(corpus)

# Create a Chroma collection
collection = client.create_collection(name="small_arxiv_ds")


# Generate unique IDs for each document

print("Corpus size: ", len(corpus))
print("--------------")
print("embeddings size: ", len(embeddings))
print("--------------")

# Add documents to the collection
addDocsChromaColl(corpus, embeddings)
print("Documents saved to Chroma!")


# Define your query
query = "sumamrize and categorize these documents into top 3 topics and list the paper titles in these topics"

results = queryChroma(query)

PROMPT = f"query is {query} and the retrievel results are {results['documents']}. please create a rag using this information"
print("prompt = ", PROMPT)

# Generate text
response = ollama.chat(model=LLM_MODEL_NAME, messages=[{"role": "user", "content": PROMPT}])
# Output the response
print("final result = ", response['message']['content'])
fd_out = open("data/lc_chroma_ollama_naive1.txt", "w")
fd_out.write(str(response['message']['content']))
fd_out.close()
