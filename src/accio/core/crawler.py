from typing import Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urlparse

from .config import CrawlerConfig


class CrawlerError(Exception):
    """Base exception for crawler errors"""

    pass


class Crawler:
    """Base crawler class with retry logic and error handling"""

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create and configure requests session"""
        session = requests.Session()
        session.headers.update(self.config.headers)
        session.headers["User-Agent"] = self.config.user_agent
        return session

    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            logger.error(f"Invalid URL format: {url}, error: {e}")
            return False

    def fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage

        Args:
            url: The URL to fetch

        Returns:
            BeautifulSoup object or None if failed

        Raises:
            CrawlerError: If the fetch operation fails after retries
        """
        if not self._validate_url(url):
            raise CrawlerError(f"Invalid URL: {url}")

        retries = 0
        while retries <= self.config.max_retries:
            try:
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()

                return BeautifulSoup(response.text, "lxml")

            except requests.RequestException as e:
                retries += 1
                if retries > self.config.max_retries:
                    logger.error(f"Failed to fetch {url} after {retries} retries: {e}")
                    raise CrawlerError(f"Failed to fetch {url}: {e}")
                logger.warning(
                    f"Retry {retries}/{self.config.max_retries} for {url}: {e}"
                )

    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text from the webpage

        Args:
            soup: BeautifulSoup object of the webpage

        Returns:
            Extracted text content
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    def close(self):
        """Close the session"""
        self.session.close()
