from accio import Accio
from loguru import logger


def main():
    # Initialize Accio
    accio = Accio()

    # Test URL
    url = "https://en.wikipedia.org/wiki/Web_crawler"

    try:
        # Fetch and process the URL
        logger.info(f"Fetching and processing: {url}")
        chunks = accio.fetch_and_process(url)

        if chunks:
            logger.info(f"Successfully processed URL. Got {len(chunks)} chunks.")

            # Print first few chunks
            for i, chunk in enumerate(chunks[:3]):
                logger.info(f"\nChunk {i + 1}:")
                logger.info("=" * 50)
                logger.info(chunk[:200] + "..." if len(chunk) > 200 else chunk)

        else:
            logger.error("Failed to process URL")

    finally:
        # Cleanup
        accio.close()


if __name__ == "__main__":
    main()
