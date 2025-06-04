from typing import List, Optional
from loguru import logger

from ..core.config import ChunkerConfig
from ..brain.models import TextProcessor


class Chunker:
    """Text chunker that splits text into overlapping chunks using sLLM"""

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self.processor = TextProcessor()

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks using sLLM

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        return self.processor.create_chunks(
            text=text,
            max_chunk_size=self.config.max_chunk_size,
            overlap_size=self.config.overlap_size,
        )
