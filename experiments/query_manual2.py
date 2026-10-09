import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.chat_models import ChatOllama as OllamaLLM  # CORRECTED IMPORT
from tqdm import tqdm

# --- Configuration ---
PDF_PATH = "docs/survival_manual.pdf"
DB_PATH = "chroma_db"
MODEL_NAME = "nomic-embed-text"

def get_vectorstore():
    """Loads or creates the vector store."""
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)

    if os.path.exists(DB_PATH):
        print("Loading existing vector store from disk...")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    else:
        print("Creating new vector store...")
        print(f"Loading document from: {PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        if not docs:
            print(f"Error: No documents loaded from {PDF_PATH}. Check the file path and ensure it's a valid PDF.")
            exit()

        print("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        print(f"Creating embeddings for {len(splits)} chunks. This will take a while...")
        vectorstore = Chroma.from_documents(
            documents=tqdm(splits, desc="Processing Chunks"),
            embedding=embedding_function,
            persist_directory=DB_PATH
        )
    return vectorstore

# --- Main Application Logic ---
print("Initializing OpContinuity...")
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()

# --- Define the Model and Prompt ---
llm = OllamaLLM(model=MODEL_NAME)  # Use the ChatOllama class
prompt_template = """Answer the following question based only on the provided context.
Think step by step. If the context does not contain the answer, say that you don't know.

<context>
{context}
</context>

Question: {input}"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# --- Create the Retrieval Chain ---
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("\nReady to answer questions. Type 'exit' to quit.")

# --- Interactive Q&A Loop ---
while True:
    question = input("\nAsk a question: ")
    if question.lower() == 'exit':
        break

    try:
        response = retrieval_chain.invoke({"input": question})
        print("\n--- Answer ---")
        print(response["answer"])
        print("--------------")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the Ollama server is running in a separate terminal with 'ollama run llama3'.")
