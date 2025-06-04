from typing import List, Optional
from loguru import logger

from .core.config import AccioConfig
from .core.crawler import Crawler, CrawlerError
from .parser.chunker import Chunker
from .memory.cache import Cache
from .brain.models import TextProcessor


class Accio:
    """Main interface for Accio web crawler"""

    def __init__(self, config: Optional[AccioConfig] = None):
        self.config = config or AccioConfig()
        self.crawler = Crawler(config.crawler if config else None)
        self.processor = TextProcessor(config.model if config else None)
        self.chunker = Chunker(config.chunker if config else None)
        self.cache = Cache(config.cache if config else None)

    def fetch_and_process(self, url: str) -> Optional[List[str]]:
        """Fetch URL and process its content into chunks

        Args:
            url: URL to fetch and process

        Returns:
            List of text chunks or None if failed
        """
        # Try to get from cache first
        cached_result = self.cache.get(url)
        if cached_result is not None:
            logger.info(f"Retrieved from cache: {url}")
            return cached_result

        try:
            # Fetch and parse the webpage
            soup = self.crawler.fetch(url)
            if not soup:
                return None

            # Extract text content
            text = self.crawler.extract_text(soup)

            # Create chunks using the model
            chunks = self.processor.create_chunks(
                text,
                max_chunk_size=self.config.chunker.max_chunk_size,
                overlap_size=self.config.chunker.overlap_size,
            )

            # Cache the result
            self.cache.set(url, chunks)

            return chunks

        except CrawlerError as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing URL {url}: {e}")
            return None

    def close(self):
        """Cleanup resources"""
        self.crawler.close()
        self.cache.clear()
