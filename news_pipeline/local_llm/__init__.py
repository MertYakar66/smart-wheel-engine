"""
Local LLM Preprocessing

Optional local LLM for cheap preprocessing tasks:
- Filtering irrelevant news
- Deduplication
- Categorization
- Basic formatting

Uses Ollama or similar local inference.
NOT for reasoning - use Claude/ChatGPT/Gemini for that.
"""

from news_pipeline.local_llm.preprocessor import LocalPreprocessor

__all__ = ["LocalPreprocessor"]
