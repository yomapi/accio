# Accio

Accio is a lightweight web crawler that, like the magical summoning spell 'Accio', fetches and processes web content based on natural language queries. It utilizes small language models (sLLM) to efficiently process and retrieve information from crawled web pages.

## Features

- Natural language based web content retrieval
- Context-window based content processing
- Memory-efficient in-memory caching
- Lightweight vector similarity search
- Optimized for single desktop usage

## Installation

```bash
pip install accio
```

## Quick Start

```python
from accio import Accio

# Initialize Accio
accio = Accio()

# Fetch and process content from a URL
result = accio.fetch("https://example.com", "What is the main topic of this page?")
print(result)
```

## Development

```bash
# Clone the repository
git clone https://github.com/yomapi/accio.git
cd accio

# Install dependencies
poetry install

# Run tests
poetry run pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
