import os
import sys
import re
from PyPDF2 import PdfReader

# --- Configuration ---
PDF_PATH = "docs/survival_manual.pdf"
CHAPTER_TITLE = "CHAPTER 7 - FIRECRAFT"
PAGES_TO_SCAN = 20  # Number of pages to search starting from firecraft chapter

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"[!] File not found: {pdf_path}")
        sys.exit(1)
    
    reader = PdfReader(pdf_path)
    all_pages = [page.extract_text() or "" for page in reader.pages]
    return all_pages

def find_chapter_start(pages, keyword):
    for idx, text in enumerate(pages):
        if keyword.lower() in text.lower():
            return idx
    return -1

def search_in_text(chunks, query):
    matches = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for idx, chunk in enumerate(chunks):
        if pattern.search(chunk):
            matches.append((idx, chunk.strip()))
    return matches

def main():
    print("==== RAW TEXT SEARCH: FM 21-76 ====")
    pages = extract_text_from_pdf(PDF_PATH)
    
    chapter_start = find_chapter_start(pages, CHAPTER_TITLE)
    if chapter_start == -1:
        print(f"[!] Could not find '{CHAPTER_TITLE}' in the document.")
        sys.exit(1)

    firecraft_section = pages[chapter_start:chapter_start + PAGES_TO_SCAN]

    print("\n[✓] Ready. Type a keyword or phrase to search. Type 'exit' to quit.")
    while True:
        query = input("\nSearch for: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        matches = search_in_text(firecraft_section, query)
        if not matches:
            print("No results found.")
        else:
            print(f"\n--- Found {len(matches)} match(es) ---")
            for idx, chunk in matches:
                print(f"\n[Page offset +{idx}]")
                print(chunk[:1000])  # Truncate to 1000 chars for readability
                print("...")

if __name__ == "__main__":
    main()
