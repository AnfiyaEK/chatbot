import rag
from rag import LANGUAGE_MODEL

import eel, ollama, chromadb
import base64, io

eel.init("web")

chroma = chromadb.PersistentClient(path = "chromadb-dev")
exists = "rfp-test" in [x.name for x in chroma.list_collections()]

if not exists:
  collection = chroma.create_collection("rfp-test")
else:
  chroma.delete_collection("rfp-test")
  collection = chroma.create_collection("rfp-test")

@eel.expose
def process_base64_file(base64_string):
  global collection 

  blob = base64.standard_b64decode(base64_string)
  stream = io.BytesIO(blob)
  text = rag.extract_text_from_pdf(stream)
  chunks = rag.create_chunks(text)

  rag.add_embeddings_to_db(collection, chunks)

  return True

@eel.expose
def process_and_answer_question(question):
  context = rag.get_context(collection, question)
  system_prompt = f"You must provide responses based on the context provided. Do not use any general, outside knowledge. Be succinct. Context: {context}" 

  print("Context given:", context)

  print("Generating answer...")
  response = ollama.chat(
    model = LANGUAGE_MODEL,
    messages=[
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': question},
    ]
  )

  print("Done generating answer.")

  return response.message.content

eel.start("main.html")

