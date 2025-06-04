from accio import Crawler
from accio.core.config import CrawlerConfig
from loguru import logger


def main():
    # Initialize crawler with default config
    crawler = Crawler(CrawlerConfig())

    # Example URL to crawl
    url = "https://en.wikipedia.org/wiki/Web_crawler"

    try:
        logger.info(f"Fetching and processing: {url}")

        # Fetch and process the content
        content = crawler.fetch_content(url)

        if content:
            print("Processed content:")
            print(content)
        else:
            logger.error("No content retrieved")

    except Exception as e:
        logger.error(f"Failed to process URL: {e}")
    finally:
        crawler.close()


if __name__ == "__main__":
    main()
