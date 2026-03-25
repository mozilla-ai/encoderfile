# Local RAG with Encoderfile + Llamafile

A fully local RAG (Retrieval-Augmented Generation) system. Give it any text file, ask it questions. Everything stays local.

- **Encoderfile** handles embedding locally. 
- **Llamafile** runs the LLM locally. 
- **NumPy** handles similarity search in memory.

This is a good fit for offline environments, sensitive documents, or anywhere you need a simple, self-contained question-answering system without cloud dependencies.

Check out the full code and instructions in [GitHub](https://github.com/mozilla-ai/encoderfile/tree/main/examples/local-rag).
