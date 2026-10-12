# RAG Document Agent

A Streamlit app for asking questions about your own documents, with retrieval and generation running entirely on your machine. You upload a PDF, Word, PowerPoint, or text file, the app indexes it locally, and a local LLM streams back an answer grounded in the most relevant passages.

## How it works

The app is a small retrieval-augmented generation (RAG) pipeline with three stages, each in its own module under `src/rag_document_agent/`.

1. **Text extraction and chunking** (`document_reader.py`). `DocumentReader` infers the file type from the extension, extracts the text (including tables in Word and PowerPoint files), normalizes whitespace, and splits the result into overlapping chunks of 1000 characters with a 150 character overlap.
2. **Embedding and retrieval** (`local_embeddings.py`). `LocalEmbedding` embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2` and stores the vectors in a LangChain `InMemoryVectorStore`. For each question it returns the 10 most similar chunks.
3. **Generation** (`local_llm.py`). `AiModel` loads `Qwen/Qwen2.5-3B-Instruct` through `transformers`, builds a prompt that restricts the model to the retrieved text, and streams the answer token by token.

`app.py` ties these together in a Streamlit chat interface. It loads the LLM once per server process and builds the index once per uploaded document, so follow-up questions only pay for retrieval and generation.

## Supported file types

| Type | Extensions |
| --- | --- |
| PDF | `.pdf` |
| Word | `.docx` |
| PowerPoint | `.pptx` |
| Plain text and Markdown | `.txt`, `.md` |

## Requirements

Before installing, make sure you have the following in place.

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Hugging Face access token, because the app logs in to Hugging Face at startup
- Enough disk space and memory for the models, which are downloaded on first run and total several gigabytes

A CUDA GPU is recommended for usable generation speed. The startup check only looks for CUDA, so on other hardware it prints a warning and continues.

## Setup

Install the dependencies and create your environment file:

```bash
uv sync
cp .env_example .env
```

Then open `.env` and set `HF_TOKEN` to your Hugging Face token.

## Running the app

With the environment configured, start the Streamlit server:

```bash
uv run rag-document-agent
```

The command is a thin wrapper around `streamlit run`, so any extra arguments are forwarded to Streamlit (for example `uv run rag-document-agent --server.port 8502`).

The first launch takes a while because the models are downloaded and loaded. Once the sidebar shows that the LLM is ready, upload a document and start asking questions. The `data/` directory contains sample 10-K filings you can use to try it out.

## Project layout

```
src/rag_document_agent/
  __init__.py           Entry point that launches the Streamlit app
  app.py                Streamlit UI and session handling
  document_reader.py    Text extraction and chunking
  local_embeddings.py   Embedding model and in-memory vector index
  local_llm.py          Local LLM loading, prompt construction, streaming
data/                   Sample documents
```

## Limitations

The current design keeps things simple, which comes with a few constraints worth knowing about.

- The index lives in memory and is tied to the browser session, so it is rebuilt whenever a document is uploaded again.
- Only one document is indexed at a time. Uploading a different file replaces the index and clears the chat.
- Each question is answered independently. Earlier turns are shown in the chat but are not passed to the model.
- Scanned PDFs without a text layer produce no text, because there is no OCR step.
