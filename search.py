from pathlib import Path

import pymupdf
from openai import OpenAI
from sentence_transformers import SentenceTransformer


pdf_path = Path(__file__).with_name("document.pdf")

# Reads OPENAI_API_KEY automatically from your environment.
openai_client = OpenAI()


def split_into_chunks(
    text: str,
    page_number: int,
    chunk_size: int = 120,
    overlap: int = 30,
):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]

        if not chunk_words:
            continue

        chunks.append(
            {
                "text": " ".join(chunk_words),
                "page": page_number,
            }
        )

    return chunks


all_chunks = []

with pymupdf.open(pdf_path) as pdf:
    print(f"PDF contains {pdf.page_count} pages.")

    for page_index, page in enumerate(pdf):
        page_text = page.get_text("text")

        page_chunks = split_into_chunks(
            text=page_text,
            page_number=page_index + 1,
        )

        all_chunks.extend(page_chunks)


if not all_chunks:
    raise ValueError(
        "No text was extracted. The PDF may be scanned or image-based."
    )


print(f"Created {len(all_chunks)} chunks.")
print("Loading embedding model...")


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


chunk_texts = [chunk["text"] for chunk in all_chunks]
chunk_embeddings = embedding_model.encode(chunk_texts)


print("\nRAG is ready.")
print("Ask questions about the PDF.")
print("Type 'exit' to stop.\n")


while True:
    question = input("Your question: ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.\n")
        continue

    question_embedding = embedding_model.encode([question])

    similarities = embedding_model.similarity(
        question_embedding,
        chunk_embeddings,
    )[0]

    results = sorted(
        zip(all_chunks, similarities.tolist()),
        key=lambda result: result[1],
        reverse=True,
    )

    top_k = 3
    top_results = results[:top_k]

    context = "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}"
        for chunk, score in top_results
    )

    print("\nGenerating an answer...\n")

    response = openai_client.responses.create(
        model="gpt-5.6-luna",
        instructions=(
            "You are a retrieval-augmented assistant. "
            "Answer the question using only the supplied PDF context. "
            "Do not invent information. "
            "If the answer is not present in the context, say that you "
            "could not find it in the document. "
            "Cite supporting pages using the format [Page 1]."
        ),
        input=(
            f"Question:\n{question}\n\n"
            f"Retrieved PDF context:\n{context}"
        ),
    )

    print("Answer:")
    print(response.output_text)

    source_pages = sorted(
        {chunk["page"] for chunk, score in top_results}
    )

    formatted_pages = ", ".join(
        f"Page {page}" for page in source_pages
    )

    print(f"\nRetrieved sources: {formatted_pages}")

    print("\nRetrieval details:")

    for chunk, score in top_results:
        print(
            f"Page {chunk['page']} "
            f"| Similarity: {score:.4f}"
        )

    print()