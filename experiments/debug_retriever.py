import os
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DB_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

# --- Main Debugging Logic ---
print("Initializing Debugger...")

if not os.path.exists(DB_PATH):
    print(f"Error: Database not found at {DB_PATH}. Please run query_manual.py first to create it.")
    exit()

print("Loading vector store...")
embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

print("Creating retriever...")
retriever = vectorstore.as_retriever()

question = "What are the key differences between a lean-to and a debris hut?"
print(f"Testing retriever with question: '{question}'")

# --- Invoke the retriever directly ---
# This bypasses the LLM and just shows us what documents are found.
retrieved_docs = retriever.invoke(question)

# --- Analyze the result ---
print("\n--- Retriever Output ---")
if not retrieved_docs:
    print("RESULT: The retriever found NO relevant documents.")
    print("This likely means the information is not in the source PDF or is described in a very different way.")
else:
    print(f"RESULT: The retriever found {len(retrieved_docs)} document(s).")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content)
        # You can also see which page it came from
        if 'page' in doc.metadata:
            print(f"Source Page: {doc.metadata['page']}")
        print("--------------------")
