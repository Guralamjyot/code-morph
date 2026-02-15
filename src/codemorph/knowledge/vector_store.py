"""
Vector Store for RAG-based Style Retrieval

Implements the two-tier retrieval strategy from Section 17.1:
- Bootstrap Layer: Pre-seed with golden reference examples
- Snowball Layer: Index verified translations during Phase 2

This ensures architectural consistency and style matching.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CodeExample:
    """A code example stored in the vector store."""

    id: str
    language: str
    code: str
    description: str
    category: str  # e.g., "error_handling", "list_comprehension", "class_definition"

    # Metadata
    source: str  # "bootstrap" or "snowball"
    verified: bool = False
    fragment_type: Optional[str] = None

    # Vector representation (will be computed)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "language": self.language,
            "code": self.code,
            "description": self.description,
            "category": self.category,
            "source": self.source,
            "verified": self.verified,
            "fragment_type": self.fragment_type,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CodeExample":
        """Create from dictionary."""
        return cls(**data)


class SimpleEmbedder:
    """
    Simple code embedder using TF-IDF-like approach.

    In production, this would use a proper code embedding model like:
    - CodeBERT
    - GraphCodeBERT
    - nomic-embed-code (via Ollama)
    """

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def embed(self, code: str) -> List[float]:
        """Generate embedding for code snippet."""
        # Simple bag-of-tokens approach
        # In production, use transformer-based model

        tokens = self._tokenize(code)

        # Create a simple embedding (128-dim)
        embedding = [0.0] * 128

        for i, token in enumerate(tokens[:128]):
            # Simple hash-based embedding
            hash_val = hash(token) % 128
            embedding[hash_val] += 1.0

        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _tokenize(self, code: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Split on whitespace and special characters
        tokens = re.findall(r'\w+|[^\w\s]', code.lower())
        return tokens

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Cosine similarity between embeddings."""
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        return dot_product  # Already normalized


class VectorStore:
    """
    In-memory vector store for code examples.

    Implements the two-tier retrieval strategy:
    - Bootstrap: Golden reference examples
    - Snowball: Verified translations from Phase 2
    """

    def __init__(self, storage_dir: Path, embedder: Optional[SimpleEmbedder] = None):
        """
        Initialize vector store.

        Args:
            storage_dir: Directory for persistent storage
            embedder: Embedding model (uses simple embedder if None)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or SimpleEmbedder()

        # In-memory index
        self.examples: Dict[str, CodeExample] = {}

        # Load existing examples
        self._load_from_disk()

    def add_example(
        self,
        example_id: str,
        language: str,
        code: str,
        description: str,
        category: str,
        source: str = "bootstrap",
        verified: bool = False,
        fragment_type: Optional[str] = None,
    ) -> CodeExample:
        """
        Add a code example to the store.

        Args:
            example_id: Unique identifier
            language: Programming language
            code: Code snippet
            description: Human-readable description
            category: Example category (e.g., "error_handling")
            source: "bootstrap" or "snowball"
            verified: Whether this is a verified translation
            fragment_type: Type of code fragment

        Returns:
            The created CodeExample
        """
        # Generate embedding
        embedding = self.embedder.embed(code)

        example = CodeExample(
            id=example_id,
            language=language,
            code=code,
            description=description,
            category=category,
            source=source,
            verified=verified,
            fragment_type=fragment_type,
            embedding=embedding,
        )

        self.examples[example_id] = example

        # Persist to disk
        self._save_example(example)

        logger.debug(f"Added example {example_id} ({source} layer)")

        return example

    def query(
        self,
        query_code: str,
        language: str,
        category: Optional[str] = None,
        top_k: int = 3,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[CodeExample, float]]:
        """
        Query for similar code examples.

        Args:
            query_code: Code to find similar examples for
            language: Target language to filter by
            category: Optional category filter
            top_k: Number of results to return
            source_filter: Filter by source ("bootstrap" or "snowball")

        Returns:
            List of (example, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query_code)

        # Filter candidates
        candidates = []
        for example in self.examples.values():
            # Filter by language
            if example.language != language:
                continue

            # Filter by category if specified
            if category and example.category != category:
                continue

            # Filter by source if specified
            if source_filter and example.source != source_filter:
                continue

            # Compute similarity
            if example.embedding:
                similarity = self.embedder.similarity(query_embedding, example.embedding)
                candidates.append((example, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return candidates[:top_k]

    def get_style_examples(
        self,
        language: str,
        category: str,
        prefer_verified: bool = True,
    ) -> List[CodeExample]:
        """
        Get style examples for a specific category.

        Args:
            language: Target language
            category: Example category
            prefer_verified: Prefer verified snowball examples

        Returns:
            List of relevant code examples
        """
        examples = []

        for example in self.examples.values():
            if example.language == language and example.category == category:
                examples.append(example)

        # Sort: verified snowball > snowball > bootstrap
        if prefer_verified:
            examples.sort(
                key=lambda e: (
                    2 if e.source == "snowball" and e.verified else
                    1 if e.source == "snowball" else
                    0
                ),
                reverse=True
            )

        return examples

    def index_verified_translation(
        self,
        fragment_id: str,
        language: str,
        code: str,
        fragment_type: str,
        category: Optional[str] = None,
    ):
        """
        Index a verified translation (Snowball Layer).

        This is called after Phase 2 successful verification.

        Args:
            fragment_id: Fragment identifier
            language: Target language
            code: Translated code
            fragment_type: Type of fragment
            category: Inferred category
        """
        # Infer category if not provided
        if category is None:
            category = self._infer_category(code, fragment_type)

        # Create description
        description = f"Verified {fragment_type} from {fragment_id}"

        # Add to store
        self.add_example(
            example_id=f"snowball_{fragment_id}",
            language=language,
            code=code,
            description=description,
            category=category,
            source="snowball",
            verified=True,
            fragment_type=fragment_type,
        )

        logger.info(f"Indexed verified translation: {fragment_id} ({category})")

    def load_bootstrap_examples(self, bootstrap_dir: Path):
        """
        Load golden reference examples (Bootstrap Layer).

        Args:
            bootstrap_dir: Directory containing JSON files with examples
        """
        bootstrap_dir = Path(bootstrap_dir)

        if not bootstrap_dir.exists():
            logger.warning(f"Bootstrap directory not found: {bootstrap_dir}")
            return

        count = 0
        for json_file in bootstrap_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Expected format: list of examples
                if isinstance(data, list):
                    for example_data in data:
                        self.add_example(
                            example_id=example_data["id"],
                            language=example_data["language"],
                            code=example_data["code"],
                            description=example_data["description"],
                            category=example_data.get("category", "general"),
                            source="bootstrap",
                            verified=True,
                        )
                        count += 1

            except Exception as e:
                logger.warning(f"Failed to load bootstrap file {json_file}: {e}")

        logger.info(f"Loaded {count} bootstrap examples")

    def _infer_category(self, code: str, fragment_type: str) -> str:
        """Infer category from code analysis."""
        code_lower = code.lower()

        # Simple heuristics
        if "try" in code_lower and ("catch" in code_lower or "except" in code_lower):
            return "error_handling"
        elif "class" in code_lower:
            return "class_definition"
        elif "interface" in code_lower:
            return "interface_definition"
        elif "for" in code_lower or "while" in code_lower:
            return "iteration"
        elif "if" in code_lower:
            return "conditional"
        elif "import" in code_lower:
            return "imports"
        else:
            return fragment_type or "general"

    def _save_example(self, example: CodeExample):
        """Save example to disk."""
        filepath = self.storage_dir / f"{example.id}.json"

        with open(filepath, 'w') as f:
            json.dump(example.to_dict(), f, indent=2)

    def _load_from_disk(self):
        """Load all examples from disk."""
        if not self.storage_dir.exists():
            return

        count = 0
        for json_file in self.storage_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                example = CodeExample.from_dict(data)
                self.examples[example.id] = example
                count += 1

            except Exception as e:
                logger.warning(f"Failed to load example {json_file}: {e}")

        if count > 0:
            logger.info(f"Loaded {count} examples from disk")

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the vector store."""
        stats = {
            "total": len(self.examples),
            "bootstrap": 0,
            "snowball": 0,
            "verified": 0,
        }

        for example in self.examples.values():
            if example.source == "bootstrap":
                stats["bootstrap"] += 1
            elif example.source == "snowball":
                stats["snowball"] += 1

            if example.verified:
                stats["verified"] += 1

        # Count by language
        by_language = {}
        for example in self.examples.values():
            by_language[example.language] = by_language.get(example.language, 0) + 1

        stats["by_language"] = by_language

        return stats
