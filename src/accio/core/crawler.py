from typing import Optional, Dict, Any, List, Tuple
import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger
from urllib.parse import urlparse
import tiktoken

from .config import CrawlerConfig
from ..brain.models import TextProcessor


class CrawlerError(Exception):
    """Base exception for crawler errors"""

    pass


class Crawler:
    """Base crawler class with retry logic and error handling"""

    # 태그별 처리 설정
    semantic_tags = {
        # 구조적 태그
        "article": {"min_length": 0, "summarize": True},
        "section": {"min_length": 0, "summarize": True},
        "main": {"min_length": 0, "summarize": True},
        # 제목 태그 (요약하지 않음)
        "h1": {"min_length": 0, "summarize": False},
        "h2": {"min_length": 0, "summarize": False},
        "h3": {"min_length": 0, "summarize": False},
        "h4": {"min_length": 0, "summarize": False},
        "h5": {"min_length": 0, "summarize": False},
        "h6": {"min_length": 0, "summarize": False},
        # 컨텐츠 태그
        "p": {"min_length": 100, "summarize": True},  # 짧은 단락은 무시
        "table": {"min_length": 0, "summarize": True},
        "ul": {"min_length": 3, "summarize": True},  # 최소 3개 항목 이상
        "ol": {"min_length": 3, "summarize": True},  # 최소 3개 항목 이상
    }

    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.session = requests.Session()
        self.processor = TextProcessor()  # Initialize text processor
        logger.info("크롤러 초기화 완료")

    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            logger.error(f"Invalid URL format: {url}, error: {e}")
            return False

    def _extract_semantic_blocks(
        self, soup: BeautifulSoup
    ) -> List[Tuple[str, str, str]]:
        """의미있는 블록 단위로 컨텐츠 추출
        Returns: List of (tag_name, context, content)
        """
        blocks = []

        def get_context(element: Tag) -> str:
            """요소의 컨텍스트(상위 헤더 등) 추출"""
            context = []
            for parent in element.parents:
                if parent.name in ["article", "section"]:
                    header = parent.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                    if header:
                        context.append(header.get_text(strip=True))
            return " > ".join(reversed(context))

        def process_table(table: Tag) -> str:
            """테이블을 구조화된 텍스트로 변환"""
            # 캡션 추출
            caption = table.find("caption")
            caption_text = (
                f"Caption: {caption.get_text(strip=True)}\n" if caption else ""
            )

            # 헤더 추출
            headers = []
            header_row = table.find("thead")
            if header_row:
                headers = [
                    th.get_text(strip=True) for th in header_row.find_all(["th", "td"])
                ]
            else:
                first_row = table.find("tr")
                if first_row:
                    headers = [
                        th.get_text(strip=True)
                        for th in first_row.find_all(["th", "td"])
                    ]

            # 데이터 행 추출
            rows = []
            for tr in table.find_all("tr")[1:] if headers else table.find_all("tr"):
                row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if any(row):  # 빈 행 제외
                    rows.append(row)

            # 테이블 텍스트 구성
            table_text = caption_text
            if headers:
                table_text += "Headers: " + " | ".join(headers) + "\n"
            table_text += "Data:\n" + "\n".join(" | ".join(row) for row in rows)

            return table_text

        def process_list(list_elem: Tag) -> str:
            """리스트를 구조화된 텍스트로 변환"""
            items = []
            for i, li in enumerate(list_elem.find_all("li", recursive=False), 1):
                text = li.get_text(strip=True)
                if list_elem.name == "ol":
                    items.append(f"{i}. {text}")
                else:
                    items.append(f"• {text}")
            return "\n".join(items)

        for tag, config in self.semantic_tags.items():
            elements = soup.find_all(tag)
            for elem in elements:
                # 컨텐츠 추출
                if tag == "table":
                    content = process_table(elem)
                elif tag in ["ul", "ol"]:
                    content = process_list(elem)
                    # 리스트 항목 수 체크
                    if len(elem.find_all("li", recursive=False)) < config["min_length"]:
                        continue
                else:
                    content = elem.get_text(strip=True)
                    # 최소 길이 체크
                    if len(content) < config["min_length"]:
                        continue

                context = get_context(elem)
                blocks.append((tag, context, content))

        return blocks

    def fetch_content(self, url: str) -> Optional[str]:
        """Fetch and process content from URL"""
        if not self._validate_url(url):
            raise CrawlerError(f"Invalid URL: {url}")

        try:
            logger.info(f"URL에서 콘텐츠 가져오는 중: {url}")
            response = self.session.get(url)
            response.raise_for_status()

            logger.info("HTML 파싱 시작")
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # 의미론적 블록 추출
            blocks = self._extract_semantic_blocks(soup)
            logger.info(f"추출된 의미론적 블록 수: {len(blocks)}")

            # 각 블록 처리
            processed_blocks = []
            for tag, context, content in blocks:
                # 태그별 처리 설정 가져오기
                config = self.semantic_tags[tag]

                # 요약이 필요없는 태그는 그대로 사용
                if not config["summarize"]:
                    processed_blocks.append(f"[{tag.upper()}] {content}")
                    continue

                # 컨텐츠가 너무 길면 분할 처리
                if len(content) > 1500:  # 토큰 제한을 고려한 문자 수
                    chunks = content.split("\n\n")  # 단락 단위로 분할
                else:
                    chunks = [content]

                # 각 청크 처리
                for chunk in chunks:
                    if not chunk.strip():
                        continue

                    prompt = f"""<instruction>다음은 웹페이지의 {tag} 요소입니다.
컨텍스트: {context if context else '없음'}
이 내용의 핵심을 간단히 요약해주세요.</instruction>

<text>
{chunk}
</text>

<response>"""

                    try:
                        response = self.processor.model(
                            prompt,
                            max_tokens=self.config.max_tokens,
                            temperature=0.1,
                        )
                        summary = response["choices"][0]["text"].strip()
                        if summary:
                            processed_blocks.append(f"[{tag.upper()}] {summary}")
                    except Exception as e:
                        logger.error(f"블록 처리 중 에러 발생: {e}")
                        continue

            result = "\n\n".join(processed_blocks) if processed_blocks else None
            logger.info("모든 블록 처리 완료")
            return result

        except requests.RequestException as e:
            logger.error(f"URL {url} 가져오기 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 에러 발생 URL {url} 처리 중: {e}")
            return None

    def process_chunk(self, text: str) -> Optional[str]:
        """Process a single chunk of text using Mistral model"""
        try:
            logger.info(f"청크 처리 시작 (길이: {len(text)})")
            # Create a summarization prompt
            prompt = f"""<instruction>Please summarize the following text, preserving the key information and main points. Make it concise but informative.</instruction>

<text>
{text}
</text>

<response>"""

            # Get response from model
            logger.info("모델에 요청 전송")
            response = self.processor.model(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=0.1,  # Low temperature for more consistent summaries
                stop=["</response>"],
            )
            logger.info("모델 응답 수신")

            summary = response["choices"][0]["text"].strip()
            return summary if summary else text.strip()

        except Exception as e:
            logger.error(f"청크 처리 중 에러 발생: {e}")
            return None

    def close(self):
        """Cleanup resources"""
        self.session.close()
        # No need to explicitly close the model as it's handled by the TextProcessor


def split_text_into_chunks(
    text: str, max_tokens: int = 1500, overlap_tokens: int = 100
) -> List[str]:
    """Split text into chunks that fit within token limit while preserving sentences.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    # First split into sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Encode sentence
        sentence_tokens = enc.encode(sentence + ".")  # Add back the period
        sentence_length = len(sentence_tokens)

        # If single sentence is too long, split it
        if sentence_length > max_tokens:
            # Split long sentence into smaller parts
            for i in range(0, len(sentence_tokens), max_tokens - overlap_tokens):
                part = enc.decode(sentence_tokens[i : i + max_tokens])
                chunks.append(part)
            continue

        # If adding this sentence would exceed max_tokens
        if current_length + sentence_length > max_tokens:
            # Store current chunk
            if current_chunk:
                chunks.append(enc.decode(current_chunk))

            # Start new chunk with overlap from previous chunk
            if current_chunk and overlap_tokens > 0:
                overlap_start = max(0, len(current_chunk) - overlap_tokens)
                current_chunk = current_chunk[overlap_start:]
                current_length = len(current_chunk)
            else:
                current_chunk = []
                current_length = 0

        # Add sentence to current chunk
        current_chunk.extend(sentence_tokens)
        current_length += sentence_length

    # Add final chunk if not empty
    if current_chunk:
        chunks.append(enc.decode(current_chunk))

    return chunks


def fetch_and_process(url: str) -> Optional[str]:
    """Fetch content from URL and process it."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Split into chunks if text is too long
        chunks = split_text_into_chunks(text)

        # Process each chunk and combine results
        processed_chunks = []
        for chunk in chunks:
            try:
                processed = process_text(chunk)  # This should be defined elsewhere
                if processed:
                    processed_chunks.append(processed)
            except Exception as e:
                logger.warning(f"Error processing chunk: {e}")
                continue

        return " ".join(processed_chunks) if processed_chunks else None

    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}")
        return None
