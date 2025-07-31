import os
import shutil # Import shutil for directory removal
from utils.document_parser import parse_document
from utils.text_splitter import chunk_text
from utils.embeddings import create_embeddings
from utils.vector_store import VectorStore
import google.generativeai as genai

# --- Setup for testing ---
TEST_DOCS_DIR = "test_docs"
# Clear and recreate the directory for a clean test run each time
if os.path.exists(TEST_DOCS_DIR):
    shutil.rmtree(TEST_DOCS_DIR) # Remove directory and all its contents
os.makedirs(TEST_DOCS_DIR)

print(f"Created/cleaned '{TEST_DOCS_DIR}' directory.")

# Sample TXT file
sample_txt_content = """
This is a sample text document. It contains some information that can be split into chunks.
The quick brown fox jumps over the lazy dog.
Dogs and cats are common household pets. They often bring joy to their owners.
Financial reports for Q1 showed significant growth in the technology sector, particularly in AI.
The stock market saw a dip, but analysts predict a strong recovery in the next quarter.
"""
with open(os.path.join(TEST_DOCS_DIR, "sample.txt"), "w") as f:
    f.write(sample_txt_content)

# Sample CSV file
sample_csv_content = """
Name,Age,City
Alice,30,New York
Bob,24,London
Charlie,35,Paris
David,28,Berlin
Eve,42,Tokyo
"""
with open(os.path.join(TEST_DOCS_DIR, "sample.csv"), "w") as f:
    f.write(sample_csv_content)

# Placeholder messages for files that need to be manually added
if not os.path.exists(os.path.join(TEST_DOCS_DIR, "sample.pdf")):
    print(f"Please place a 'sample.pdf' in the '{TEST_DOCS_DIR}' directory to test PDF parsing.")
if not os.path.exists(os.path.join(TEST_DOCS_DIR, "sample.docx")):
    print(f"Please place a 'sample.docx' in the '{TEST_DOCS_DIR}' directory to test DOCX parsing.")
if not os.path.exists(os.path.join(TEST_DOCS_DIR, "sample.pptx")):
    print(f"Please place a 'sample.pptx' in the '{TEST_DOCS_DIR}' directory to test PPTX parsing.")

# --- Main test logic ---
if __name__ == "__main__":
    print("\n--- Starting Core Components Test ---")

    all_processed_chunks = []
    
    print(f"\n--- 1. Parsing and Chunking Documents from '{TEST_DOCS_DIR}' ---")
    
    # Iterate through all files in the test directory
    for filename in os.listdir(TEST_DOCS_DIR):
        file_path = os.path.join(TEST_DOCS_DIR, filename)
        
        if os.path.isdir(file_path): # Skip subdirectories
            continue
            
        file_extension = filename.split('.')[-1].lower()
        
        supported_types = ["pdf", "docx", "csv", "pptx", "txt", "md"] # Add other types if your parser supports them
        if file_extension not in supported_types:
            print(f"Skipping unsupported file: {filename}")
            continue

        print(f"Processing {filename} ({file_extension})...")
        content = parse_document(file_path, file_extension)
        if content:
            chunks = chunk_text(content, chunk_size=500, chunk_overlap=100) # Adjust chunk size/overlap as needed
            all_processed_chunks.extend(chunks)
            print(f"  Extracted {len(content)} characters and created {len(chunks)} chunks.")
        else:
            print(f"  Could not extract content from {filename}.")

    if not all_processed_chunks:
        print("\nNo chunks were successfully processed. Cannot proceed with embeddings/vector store. Ensure files exist and are parsable.")
        exit()

    print(f"\nTotal processed chunks: {len(all_processed_chunks)}")
    print(f"First chunk example: '{all_processed_chunks[0][:150]}...'")

    print("\n--- 2. Creating Embeddings with Gemini API ---")
    # create_embeddings internally handles 'retrieval_document' task type
    embeddings_list = create_embeddings(all_processed_chunks)

    if not embeddings_list:
        print("\nFailed to create embeddings. Exiting.")
        exit()

    # --- Start of code that was previously misindented ---
    print(f"Successfully created {len(embeddings_list)} embeddings.")
    if embeddings_list: # This check is redundant after the previous 'if not embeddings_list: exit()'
        print(f"Embedding dimension: {len(embeddings_list[0])}") # Should be 768

    print("\n--- 3. Initializing and Populating Vector Store ---")
    # Initialize VectorStore with the correct embedding dimension
    vector_store = VectorStore(embedding_dim=768)
    vector_store.add_vectors(embeddings_list, all_processed_chunks)
    print(f"Vector store now contains {vector_store.index.ntotal} vectors.")

    print("\n--- 4. Testing Semantic Search ---")
    query = "What did the financial reports show regarding sector growth?"
    print(f"Query: '{query}'")

    query_embedding = None # Initialize query_embedding
    try:
        # Embed the query with task_type="retrieval_query"
        query_embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=[query], # Query content needs to be in a list
            task_type="retrieval_query"
        )
        # Extract the single embedding vector from the response
        query_embedding = query_embedding_response['embedding'][0]
    except Exception as e:
        print(f"Failed to embed query with Gemini API: {e}")
        # Allow the script to continue to test graceful handling of missing query_embedding
        pass

    retrieved_results = [] # Initialize retrieved_results
    if query_embedding: # Only proceed to search if query embedding was successful
        retrieved_results = vector_store.search_vectors(query_embedding, k=3)
    else:
        print("Query embedding failed, skipping semantic search.")

    if retrieved_results: # Now this check will always work without NameError
        print("\nRetrieved Top 3 Results:")
        for i, (chunk_text, similarity) in enumerate(retrieved_results):
            # For L2 distance, lower similarity (distance) is better
            print(f"  Result {i+1} (L2 Distance: {similarity:.4f}):")
            print(f"    {chunk_text[:200]}...") # Print first 200 chars
    else:
        print("\nNo relevant results found or query embedding failed.")
        
    print("\n--- Core Components Test Finished ---")

    print("\n--- 5. Testing Vector Store Persistence (Save/Load) ---")
    store_base_path = "my_rag_store" # Define a base path for saving index and texts

    # Ensure vector_store exists before trying to save
    if 'vector_store' in locals() and vector_store.index.ntotal > 0:
        vector_store.save_index(store_base_path)

        new_vector_store = VectorStore(embedding_dim=768)
        new_vector_store.load_index(store_base_path)
        print(f"Loaded store total vectors: {new_vector_store.index.ntotal}")

        # Test search on loaded store - ensure query_embedding is available
        if query_embedding:
            retrieved_results_loaded = new_vector_store.search_vectors(query_embedding, k=3)
            print("\nRetrieved Top 3 Results from Loaded Store:")
            if retrieved_results_loaded:
                for i, (text, distance) in enumerate(retrieved_results_loaded):
                    print(f"  Result {i+1} (Distance: {distance:.4f}):")
                    print(f"    Chunk: {text[:200]}...")
            else:
                print("No results found in loaded vector store.")
        else:
            print("Query embedding failed previously, skipping search on loaded store.")
    else:
        print("Skipping save/load test as vector store was not populated.")

    # Clean up saved FAISS files after test
    if os.path.exists(f"{store_base_path}.faiss"):
        os.remove(f"{store_base_path}.faiss")
        print(f"Removed {store_base_path}.faiss")
    if os.path.exists(f"{store_base_path}_texts.npy"):
        os.remove(f"{store_base_path}_texts.npy")
        print(f"Removed {store_base_path}_texts.npy")

    print("\n--- Core Components Test Complete ---")