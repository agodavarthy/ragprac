import re
import sys
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


LLM_MODEL_NAME= "llama2"
SNT_TRF_MODEL_NAME="all-MiniLM-L6-v2"

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
    return text

#Step 2: create chunks
def _calculate_cosine_distances(embeddings):
    # Calculate the cosine distance (1 - cosine similarity) between consecutive embeddings.
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances

def _combine_sentences(sentences):
    # Create a buffer by combining each sentence with its previous and next sentence to provide a wider context.
    combined_sentences = []
    for i in range(len(sentences)):
        combined_sentence = sentences[i]
        if i > 0:
            combined_sentence = sentences[i-1] + ' ' + combined_sentence
        if i < len(sentences) - 1:
            combined_sentence += ' ' + sentences[i+1]
        combined_sentences.append(combined_sentence)
    return combined_sentences


def _split_sentences(text):
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sentences

def chunk_text(text):
    # Split the input text into individual sentences.
    single_sentences_list = _split_sentences(text)

    # Combine adjacent sentences to form a context window around each sentence.
    combined_sentences = _combine_sentences(single_sentences_list)

    # Convert the combined sentences into vector representations using a neural network model.
    embeddings = generate_embeddings(combined_sentences, SNT_TRF_MODEL_NAME)

    # Calculate the cosine distances between consecutive combined sentence embeddings to measure similarity.
    distances = _calculate_cosine_distances(embeddings)

    # Determine the threshold distance for identifying breakpoints based on the 80th percentile of all distances.
    breakpoint_percentile_threshold = 80
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    # Find all indices where the distance exceeds the calculated threshold, indicating a potential chunk breakpoint.
    indices_above_thresh = [i for i, distance in enumerate(distances) if distance > breakpoint_distance_threshold]
    # Initialize the list of chunks and a variable to track the start of the next chunk.
    chunks = []
    start_index = 0
    # Loop through the identified breakpoints and create chunks accordingly.
    for index in indices_above_thresh:
        chunk = ' '.join(single_sentences_list[start_index:index+1])
        chunks.append(chunk)
        start_index = index + 1

    # If there are any sentences left after the last breakpoint, add them as the final chunk.
    if start_index < len(single_sentences_list):
        chunk = ' '.join(single_sentences_list[start_index:])
        chunks.append(chunk)

    # Return the list of text chunks.
    return chunks


# Step 3: Generate embeddings
def generate_embeddings(text, SNT_TRF_MODEL_NAME):
    model = SentenceTransformer(SNT_TRF_MODEL_NAME)
    return model.encode(text)

# Step 3: Compute cosine similarity (between a vector and a matrix)
def compute_cosine_similarity(vector, matrix):
    # Calculate the dot product between the vector and each row in the matrix
    dot_product = matrix @ vector
    
    # Calculate the norms (magnitudes) of the vector and each row in the matrix
    vector_norm = np.linalg.norm(vector)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    cosine_similarity = dot_product / (matrix_norm * vector_norm + epsilon)
    
    return cosine_similarity

# Example usage
text = extract_text_from_pdf("./data/inputdata_rag1Nov1.pdf")
chunks = chunk_text(text)
print("#chunks = ", len(chunks))
chunk_embs = generate_embeddings(chunks, SNT_TRF_MODEL_NAME)
query = "what are the top topics in this document?"
query_embs = generate_embeddings(query, SNT_TRF_MODEL_NAME)
cos_vals = (compute_cosine_similarity(query_embs, chunk_embs))
print(cos_vals)

cos_sort_args = np.argsort(cos_vals)[::-1]
context_text = "\n\n - -\n\n".join([chunks[ind] for ind in cos_sort_args])

PROMPT = f"query is {query} and the retrievel results are {context_text}. please create a rag using this information"
print("prompt = ", PROMPT)


# Generate text
response = ollama.chat(model=LLM_MODEL_NAME, messages=[{"role": "user", "content": PROMPT}])
print(response)
print("type of response = ", type(response))
# Output the response
print("final result = ", response['message']['content'])
