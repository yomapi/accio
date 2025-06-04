from pydantic import BaseModel, Field
from typing import Dict, Optional
from pathlib import Path


class CrawlerConfig(BaseModel):
    """Configuration for the crawler"""

    user_agent: str = Field(
        default="Accio/1.0 (+https://github.com/yomapi/accio)",
        description="User agent string for the crawler",
    )
    timeout: int = Field(default=30, description="Timeout for HTTP requests in seconds")
    max_retries: int = Field(
        default=3, description="Maximum number of retries for failed requests"
    )
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
        description="Default HTTP headers",
    )


class ChunkerConfig(BaseModel):
    """Configuration for the text chunker"""

    max_chunk_size: int = Field(
        default=512, description="Maximum size of each text chunk"
    )
    overlap_size: int = Field(
        default=50, description="Number of tokens to overlap between chunks"
    )
    min_chunk_size: int = Field(
        default=100, description="Minimum size of each text chunk"
    )


class CacheConfig(BaseModel):
    """Configuration for the cache"""

    max_size: int = Field(default=1000, description="Maximum number of items in cache")
    ttl: int = Field(
        default=3600, description="Time to live for cache items in seconds"
    )


class ModelConfig(BaseModel):
    """Configuration for the LLM"""

    model_key: str = Field(default="tiny", description="Which model variant to use")
    model_path: Optional[str] = Field(
        default=None, description="Optional custom path to model file"
    )
    num_threads: int = Field(default=4, description="Number of CPU threads to use")
    context_size: int = Field(default=2048, description="Model context window size")
    batch_size: int = Field(
        default=512, description="Number of tokens to process in parallel"
    )


class AccioConfig(BaseModel):
    """Main configuration for Accio"""

    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    debug: bool = Field(default=False)
