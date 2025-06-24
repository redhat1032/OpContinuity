from langchain_community.document_loaders import PyPDFDirectoryLoader

DOCS_PATH = "docs"

print(f"Attempting to load all PDFs from '{DOCS_PATH}' directory...")

loader = PyPDFDirectoryLoader(DOCS_PATH)
docs = loader.load()

if not docs:
    print("\n--- RESULT ---")
    print("Loader did not find or could not extract text from any documents.")
    print("Please ensure there are text-based PDFs in the 'docs' folder.")
    print("----------------")
else:
    print(f"\n--- RESULT: Found {len(docs)} pages across all documents. ---")
    # To avoid printing everything, we'll just show the first page of each unique file.
    processed_files = []
    for doc in docs:
        source_file = doc.metadata.get('source')
        if source_file not in processed_files:
            processed_files.append(source_file)
            print(f"\n--- Checking content from: {source_file} ---")
            print(doc.page_content[:300].strip()) # Print the first 300 chars
            print("--------------------------------------------------")
