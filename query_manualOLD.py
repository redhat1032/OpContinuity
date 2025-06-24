import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. Load the Document ---
# This loads the PDF and splits it into individual pages.
print("Loading document...")
loader = PyPDFLoader("docs/survival_manual.pdf")
docs = loader.load()

# --- 2. Split the Document into Chunks ---
# This splits the pages into smaller text chunks for the model to process.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 3. Create Embeddings and Store in Chroma DB ---
# This uses the Ollama Llama-3 model to create numerical representations (embeddings)
# of each text chunk and stores them in a local vector database (Chroma).
print("Creating embeddings and vector store...")
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# --- 4. Define the Model and Prompt Template ---
# This sets up the LLM we'll be using and defines how we'll prompt it.
llm = ollama.Llama3() # Initialize Llama-3 model from Ollama
prompt_template = """Answer the following question based only on the provided context.
Think step by step. If the context does not contain the answer, say that you don't know.

<context>
{context}
</context>

Question: {input}"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# --- 5. Create the Retrieval Chain ---
# This chain does two things:
# 1. Takes a question, finds relevant text chunks from the vector database (the "retriever").
# 2. "Stuffs" those chunks into the prompt template and sends it to the LLM.
print("Creating retrieval chain...")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 6. Ask a Question ---
print("Ready to answer questions.")
question = "What are the three types of fires I can build?"
print(f"Question: {question}")

response = retrieval_chain.invoke({"input": question})

print("\n--- Answer ---")
print(response["answer"])
print("--------------")
