from PyPDF2 import PdfReader
import ollama
import numpy as np

EMBEDDING_MODEL = 'all-minilm:l6-v2'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest'

def extract_text_from_pdf(stream):
  reader = PdfReader(stream)
  parts = []

  def extract_text_from_page(page):
    # Removes text from the header and footer (except one line in the header)
    text_parts = []

    def visitor_body(text, cm, tm, fontDict, fontSize):
      y = tm[5]
      if y > 70 and y < 680:
        text_parts.append(text)

    page.extract_text(visitor_text=visitor_body)

    return " ".join(text_parts)

  for page in reader.pages:
    parts.append(extract_text_from_page(page))

  parts = [s for s in parts if s.strip()]

  return parts


def create_chunks(text):
  chunk_size = 50
  chunks = []
  
  for i in range(0, len(text), chunk_size - 20):
    chunks.append(text[i : i + chunk_size])
  chunks_ = []

  for chunk in chunks:
    chunks_.append([" ".join(chunk)])

  return chunks_


def create_embedding_for_chunk(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  
  return (embedding, chunk)


def add_embeddings_to_db(collection, chunks):
  print("Number of chunks:", len(chunks))
  count = 0

  for i, chunk in enumerate(chunks):
    doc_id = f"chunk_{i}"
    embedding, chunk = create_embedding_for_chunk(chunk)

    collection.add(
      documents=[" ".join(chunk)],
      ids=[doc_id],
      embeddings=[embedding]
    )

    count += 1
    print(f"Processed chunk {count}")


def compute_similarity(embedding, collection, top_k):
  results = collection.query(
    query_embeddings=[embedding],
    n_results=top_k,
    include=['documents', 'distances']
  )

  similarities = []
  for doc, dist in zip(results['documents'][0], results['distances'][0]):
    similarity = 1 / (1 + dist)  # convert distance to similarity
    similarities.append((similarity, doc))
  similarities = sorted(similarities, reverse=True)

  return similarities


def embed_question(question):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=question)['embeddings'][0]

  return embedding


def get_k_most_similar_chunks(collection, question, k):
  embedding = embed_question(question)
  similarities = compute_similarity(embedding, collection, k)

  return similarities[:k]


def get_context(collection, question, k=10):
  most_similar = get_k_most_similar_chunks(collection, question, k)
  context = ""

  print(most_similar)

  for i in most_similar:
    context += i[1] + "\n"

  return context

