from typing import List, Optional
from pathlib import Path
import re
import requests
from tqdm import tqdm
import hashlib
from pydantic import BaseModel
from loguru import logger
from llama_cpp import Llama

from ..core.config import ModelConfig

MISTRAL_MODELS = {
    "tiny": {
        "name": "mistral-7b-instruct-v0.2-q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2-q4_K_M.gguf",
        "size": 4_000_000_000,  # Approximate size in bytes
    }
}


def verify_file(file_path: Path, expected_size: int) -> bool:
    """Verify that the file is a valid GGUF file"""
    if not file_path.exists():
        return False

    # Check file size
    if file_path.stat().st_size < expected_size * 0.9:  # Allow some tolerance
        return False

    # Check GGUF magic number
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
            return magic == b"GGUF"
    except Exception as e:
        logger.error(f"Error verifying file: {e}")
        return False


def download_model(model_key: str = "tiny", save_dir: Optional[Path] = None) -> Path:
    """Download the Mistral model if not already present

    Args:
        model_key: Which model variant to download
        save_dir: Where to save the model. If None, uses default location

    Returns:
        Path to the downloaded model file
    """
    if model_key not in MISTRAL_MODELS:
        raise ValueError(
            f"Unknown model key: {model_key}. Available: {list(MISTRAL_MODELS.keys())}"
        )

    model_info = MISTRAL_MODELS[model_key]

    # Determine save location
    if save_dir is None:
        save_dir = Path.home() / ".cache/accio/models"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / model_info["name"]

    # Check if already downloaded and valid
    if model_path.exists() and verify_file(model_path, model_info["size"]):
        logger.info(f"Model already downloaded and verified: {model_path}")
        return model_path
    elif model_path.exists():
        logger.warning(
            f"Invalid or incomplete model found, re-downloading: {model_path}"
        )
        model_path.unlink()

    # Download the model
    logger.info(f"Downloading model to: {model_path}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
    }

    try:
        with requests.get(model_info["url"], stream=True, headers=headers) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", model_info["size"]))

            with open(model_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=f"Downloading {model_info['name']}",
                ) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        size = f.write(data)
                        pbar.update(size)
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        if model_path.exists():
            model_path.unlink()
        raise

    # Verify downloaded file
    if not verify_file(model_path, model_info["size"]):
        model_path.unlink()
        raise RuntimeError("Downloaded file is invalid or corrupted")

    return model_path


class ChunkingPrompt(BaseModel):
    """Base prompt for text chunking"""

    text: str
    max_chunk_size: int
    overlap_size: int

    def to_prompt(self) -> str:
        return f"""<instruction>Split the following text into meaningful chunks. Each chunk should:
- Be a coherent unit of information
- Have maximum {self.max_chunk_size} words
- Preserve complete sentences where possible
- Maintain context between chunks

Return the chunks as a numbered list, one chunk per line.</instruction>

<text>
{self.text}
</text>

<response>"""


class TextProcessor:
    """Text processor using Mistral model"""

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize text processor with Mistral model

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model_path = (
            Path(self.config.model_path)
            if self.config.model_path
            else self._find_or_download_model(self.config.model_key)
        )
        logger.info(f"Loading Mistral model from: {self.model_path}")

        self.model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.config.context_size,
            n_threads=self.config.num_threads,
            n_batch=self.config.batch_size,
        )

    def _find_or_download_model(self, model_key: str) -> Path:
        """Find existing model or download if not found"""
        # Check standard locations first
        standard_locations = [
            Path.home() / ".cache/accio/models",
            Path("/usr/local/share/accio/models"),
            Path("models"),  # Relative to current directory
        ]

        model_name = MISTRAL_MODELS[model_key]["name"]

        for location in standard_locations:
            model_path = location / model_name
            if model_path.exists():
                return model_path

        # If not found, download
        return download_model(model_key)

    def _parse_chunks(self, response: str) -> List[str]:
        """Parse numbered list from model response"""
        # Remove any leading/trailing whitespace and split into lines
        lines = response.strip().split("\n")
        chunks = []

        for line in lines:
            # Remove numbering and leading/trailing whitespace
            cleaned = re.sub(r"^\d+[\)\.]\s*", "", line).strip()
            if cleaned:
                chunks.append(cleaned)

        return chunks

    def create_chunks(
        self, text: str, max_chunk_size: int = 512, overlap_size: int = 50
    ) -> List[str]:
        """Split text into chunks using Mistral

        Args:
            text: Input text to split
            max_chunk_size: Maximum size of each chunk
            overlap_size: Number of words to overlap between chunks

        Returns:
            List of text chunks
        """
        prompt = ChunkingPrompt(
            text=text, max_chunk_size=max_chunk_size, overlap_size=overlap_size
        )

        # Generate response from model
        response = self.model(
            prompt.to_prompt(),
            max_tokens=self.config.context_size,
            temperature=0.1,  # Low temperature for more deterministic output
            stop=["</response>"],  # Stop generation at this token
        )

        # Parse chunks from response
        chunks = self._parse_chunks(response["choices"][0]["text"])

        # Ensure we have at least one chunk
        if not chunks:
            logger.warning("Model returned no chunks, falling back to simple splitting")
            words = text.split()
            for i in range(0, len(words), max_chunk_size - overlap_size):
                chunk = " ".join(words[i : i + max_chunk_size])
                chunks.append(chunk)

        return chunks
