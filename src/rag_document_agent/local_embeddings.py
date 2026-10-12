"""
Local sentence embeddings and vector search via LangChain.

Wraps LangChain's HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2,
running fully locally through the sentence-transformers backend) and its
InMemoryVectorStore to build a per-document semantic index for RAG retrieval.
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


class LocalEmbedding:
    """Local embedding + vector store wrapper backed by LangChain."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialise the embedding model and an empty vector store.

        Args:
            model_name: HuggingFace model ID to load locally.
        """
        self.model_name = model_name

        print(f"[LocalEmbedding] Loading {self.model_name}...")
        # normalize_embeddings=True makes cosine similarity scores comparable
        # to the L2-normalized vectors the rest of the app previously assumed.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[LocalEmbedding] Model ready.")

        self.store = InMemoryVectorStore(self.embeddings)

    def build_index(self, chunks: list[str]) -> None:
        """
        Embed all chunks and add them to the vector store.

        LangChain batches the embedding calls internally, so this is as
        efficient as the previous manual batch-embed step.

        Args:
            chunks: List of text strings to index.
        """
        self.store.add_texts(chunks)
        print(f"[LocalEmbedding] Indexed {len(chunks)} chunks.")

    def search(self, question: str, k: int = 3) -> list[tuple[dict, float]]:
        """
        Find the k chunks most semantically similar to the question.

        Args:
            question: Natural-language question string.
            k:        Number of results to return (default 3).

        Returns:
            List of (document_dict, score) tuples, sorted by descending
            cosine similarity (higher = more similar).
        """
        results = self.store.similarity_search_with_score(question, k=k)
        return [({"content": doc.page_content}, score) for doc, score in results]

    def get_context(self, question: str, k: int = 3) -> str:
        """
        Retrieve the most relevant chunks for a question as a single string.

        Intended for RAG: pass the returned string directly as context
        to your LLM prompt alongside the user's question.

        Args:
            question: Natural-language question string.
            k:        Number of chunks to include (default 3).

        Returns:
            A single string with the k most relevant chunks
            joined by a separator, ready to embed in a prompt.
        """
        docs = self.store.similarity_search(question, k=k)
        return "\n\n---\n\n".join(doc.page_content for doc in docs)


###############
# Example usage
###############

# from rag_document_agent.document_reader import DocumentReader
#
# embedder = LocalEmbedding()
#
# document_reader = DocumentReader()
# chunks = document_reader.get_chunks()
# embedder.build_index(chunks)

# results = embedder.search("what is this document all about", k=10)
# for doc, score in results:
#     print(f"Score: {score:.4f}")
#     print(f"Chunk: {doc['content'][:300]}")
#     print("-" * 80)
# print(embedder.get_context("what is this document all about", 10))
