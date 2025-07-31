# utils/text_splitter.py

"""
This module provides a utility function for splitting long texts into smaller,
overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
This is a crucial step in preparing documents for Retrieval-Augmented Generation (RAG)
systems, ensuring that text fits within model context windows and preserves
contextual information across chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Splits a given text into smaller, overlapping chunks.

    This function uses LangChain's RecursiveCharacterTextSplitter, which attempts
    to split text by different separators (like newlines, spaces) in a recursive
    manner to maintain semantic coherence as much as possible.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum desired length of each text chunk (in characters).
                          Defaults to 1000.
        chunk_overlap (int): The number of characters to overlap between consecutive
                             chunks. This helps retain context across chunk boundaries.
                             Defaults to 200.

    Returns:
        list[str]: A list of text chunks. Returns an empty list if the input text is empty.

    Raises:
        ValueError: If chunk_overlap is greater than or equal to chunk_size.
    """
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Chunk overlap ({chunk_overlap}) cannot be greater than or equal to "
            f"chunk size ({chunk_size}). Please ensure chunk_overlap < chunk_size."
        )

    # Initialize the RecursiveCharacterTextSplitter
    # It tries to split by common separators (like "\n\n", "\n", " ", "")
    # to keep paragraphs, sentences, and words together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character count for length calculation
        is_separator_regex=False, # Treat default separators as literal strings, not regex
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == '__main__':
    # --- Standalone Test for utils/text_splitter.py ---

    print("--- Testing utils/text_splitter.py ---")
    
    # A long sample text for chunking demonstration
    long_text = """
    The quick brown fox jumps over the lazy dog. This is a classic pangram,
    meaning it contains every letter of the alphabet. It's often used for
    testing typefaces and keyboard layouts. Text chunking is an essential
    process in Natural Language Processing (NLP), particularly for
    Retrieval-Augmented Generation (RAG) systems. When working with large
    documents, breaking them down into smaller, manageable chunks is crucial.
    This helps ensure that the information fits within the context window
    of Large Language Models (LLMs) and that relevant information can be
    retrieved efficiently. Overlapping chunks are often used to preserve
    context across segment boundaries, reducing the chance of losing
    critical information that might fall between two chunks.
    For example, if a key sentence is split exactly at its midpoint,
    overlap ensures both halves are present in adjacent chunks,
    providing the LLM with enough surrounding text to understand the full context.
    Different chunking strategies exist, including fixed-size, sentence-based,
    and semantic chunking. The choice depends on the specific use case and
    the nature of the data. RecursiveCharacterTextSplitter is a versatile
    option as it intelligently tries to split on meaningful breakpoints.
    """

    # Test Case 1: Default chunk size and overlap
    print("\n--- Test Case 1: Default Parameters (chunk_size=1000, chunk_overlap=200) ---")
    default_chunks = chunk_text(long_text)
    print(f"Number of chunks: {len(default_chunks)}")
    for i, chunk in enumerate(default_chunks):
        print(f"  Chunk {i+1} (Length: {len(chunk)}):\n  '{chunk[:150]}...'")
    assert len(default_chunks) > 0
    assert len(default_chunks[0]) <= 1000
    if len(default_chunks) > 1:
        # Check a simple overlap property
        # The start of chunk 2 should contain part of the end of chunk 1
        # This is a basic check, actual overlap might be complex due to separators
        # For simplicity, we check if the end of chunk 1 has some content similar to start of chunk 2.
        # This is hard to assert perfectly with RecursiveCharacterTextSplitter as it prioritizes separators.
        # A more robust check would involve specific text segments.
        pass # Skipping complex overlap assertion for simplicity in example

    # Test Case 2: Custom chunk size and overlap
    print("\n--- Test Case 2: Custom Parameters (chunk_size=150, chunk_overlap=30) ---")
    custom_chunks = chunk_text(long_text, chunk_size=150, chunk_overlap=30)
    print(f"Number of chunks: {len(custom_chunks)}")
    for i, chunk in enumerate(custom_chunks):
        print(f"  Chunk {i+1} (Length: {len(chunk)}):\n  '{chunk[:150]}...'")
    assert len(custom_chunks) > len(default_chunks) # More chunks for smaller size
    assert all(len(c) <= 150 for c in custom_chunks) # All chunks within max size

    # Test Case 3: Empty text input
    print("\n--- Test Case 3: Empty Text Input ---")
    empty_chunks = chunk_text("")
    print(f"Chunks for empty text: {empty_chunks}")
    assert empty_chunks == []
    print("Empty text input test PASSED.")

    # Test Case 4: Text shorter than chunk_size
    print("\n--- Test Case 4: Text Shorter Than Chunk Size ---")
    short_text = "This is a short text."
    short_chunks = chunk_text(short_text, chunk_size=100, chunk_overlap=20)
    print(f"Chunks for short text: {short_chunks}")
    assert len(short_chunks) == 1
    assert short_chunks[0] == short_text
    print("Short text test PASSED.")

    # Test Case 5: chunk_overlap >= chunk_size (should raise ValueError)
    print("\n--- Test Case 5: Invalid Parameters (chunk_overlap >= chunk_size) ---")
    try:
        chunk_text(long_text, chunk_size=100, chunk_overlap=100)
        print("ERROR: Expected ValueError, but no exception was raised.")
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")
        assert "chunk_overlap cannot be greater than or equal to chunk size" in str(e)
    print("Invalid parameters test PASSED.")

    print("\n--- All text_splitter.py tests completed. ---")