from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from compact_source_data import (
    DEFAULT_SOURCE_TABLE,
    iter_source_records,
    strategy_label_to_code,
)

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing dependency for semantic similarity.\n"
        "Install with:\n"
        "    python -m pip install -r requirements-embedding-cache.txt"
    ) from exc

try:
    import tiktoken
    from openai import OpenAI
except ImportError:
    tiktoken = None
    OpenAI = None


EMBEDDING_MODEL = "text-embedding-3-large"

# API settings are environment variables so this script is safe to publish.
# Leave OPENAI_BASE_URL empty to use the OpenAI SDK's default endpoint.
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

SCRIPT_DIR = Path(__file__).resolve().parent

EMBEDDING_CACHE_DIR = SCRIPT_DIR / "embedding_cache"
CACHE_WORKERS = max(1, min(10, os.cpu_count() or 1))
ENABLE_EMBEDDING_DISK_CACHE = True
MAX_MODEL_TOKENS = 512
EMBEDDING_CACHE_VERSION = "v1"
OPENAI_EMBEDDING_DIMENSIONS: Optional[int] = None
OPENAI_EMBEDDING_RETRIES = 5
OPENAI_EMBEDDING_RETRY_BASE_SECONDS = 1.0
STRIP_JSON_BLOCKS_FROM_OUTPUT_TXT = False

SELECTED_SOURCE_MODELS = [
    "llama-4-scout",
    "gpt-4o",
    "gemini-3.1-flash-lite",
    "deepseek-v3.2-think",
    "grok-4.3",
]

CACHE_SOURCE_MODELS = "all"

STRATEGY_PATHS = {name: {} for name in [
    "Strategy 0", "Strategy 1", "Strategy 2", "Strategy 3",
    "Strategy 1_1", "Strategy 2_1", "Strategy 3_1",
]}


@dataclass(frozen=True)
class DocumentEmbedding:
    document_embedding: torch.Tensor
    chunk_embeddings: list[torch.Tensor]
    chunk_weights: list[int]
    chunk_texts: list[str]


@dataclass(frozen=True)
class SimilarityScores:
    s_global: float
    s_coverage: float
    s_final: float


@dataclass(frozen=True)
class EncodingJob:
    text: str
    label: str


@dataclass
class CacheStats:
    strategy_dirs: int = 0
    round_dirs: int = 0
    output_txt_seen: int = 0
    valid_pairs: int = 0
    skipped_unmatched_output: int = 0
    skipped_empty_mechanistic_explanation: int = 0
    skipped_empty_output_txt: int = 0
    skipped_duplicate_this_run: int = 0
    skipped_existing_cache: int = 0
    would_encode: int = 0
    newly_encoded: int = 0
    errors: int = 0


class CacheKeyOnlyEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model_max_length = MAX_MODEL_TOKENS
        self.content_tokens_per_chunk = MAX_MODEL_TOKENS
        cache_model_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
        self.cache_dir = EMBEDDING_CACHE_DIR / cache_model_dir
        self.document_embedding_cache_dir = self.cache_dir / "documents"

    def cache_key(self, text: str) -> str:
        payload = {
            "version": EMBEDDING_CACHE_VERSION,
            "model": self.model_name,
            "dimensions": OPENAI_EMBEDDING_DIMENSIONS,
            "model_max_length": self.model_max_length,
            "content_tokens_per_chunk": self.content_tokens_per_chunk,
            "text": text,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class OpenAIEmbedder:
    def __init__(self, model_name: str) -> None:
        if OpenAI is None or tiktoken is None:
            raise SystemExit(
                "Missing OpenAI embedding dependency.\n"
                "Install with:\n"
                "    python -m pip install -r requirements-embedding-cache.txt"
            )
        self.model_name = model_name
        api_key = OPENAI_API_KEY.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "Missing OpenAI API key. Set the OPENAI_API_KEY environment variable."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        self.client = OpenAI(**client_kwargs)
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.model_max_length = MAX_MODEL_TOKENS
        self.content_tokens_per_chunk = MAX_MODEL_TOKENS
        self.chunk_embedding_cache: dict[str, torch.Tensor] = {}
        self.document_cache: dict[str, DocumentEmbedding] = {}
        self.chunk_cache: dict[str, list[tuple[str, int]]] = {}
        self.cache_lock = threading.RLock()
        self.key_locks: dict[str, threading.Lock] = {}
        cache_model_dir = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
        self.cache_dir = EMBEDDING_CACHE_DIR / cache_model_dir
        self.chunk_embedding_cache_dir = self.cache_dir / "chunks"
        self.document_embedding_cache_dir = self.cache_dir / "documents"

    def cache_key(self, text: str) -> str:
        payload = {
            "version": EMBEDDING_CACHE_VERSION,
            "model": self.model_name,
            "dimensions": OPENAI_EMBEDDING_DIMENSIONS,
            "model_max_length": self.model_max_length,
            "content_tokens_per_chunk": self.content_tokens_per_chunk,
            "text": text,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def lock_for_key(self, key: str) -> threading.Lock:
        with self.cache_lock:
            key_lock = self.key_locks.get(key)
            if key_lock is None:
                key_lock = threading.Lock()
                self.key_locks[key] = key_lock
            return key_lock

    def load_chunk_embedding_from_disk(self, key: str) -> Optional[torch.Tensor]:
        if not ENABLE_EMBEDDING_DISK_CACHE:
            return None
        path = self.chunk_embedding_cache_dir / f"{key}.pt"
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu")
            embedding = payload["embedding"]
        except Exception:
            return None
        if not isinstance(embedding, torch.Tensor):
            return None
        return embedding.float().cpu()

    def save_chunk_embedding_to_disk(
        self,
        key: str,
        text: str,
        embedding: torch.Tensor,
    ) -> None:
        if not ENABLE_EMBEDDING_DISK_CACHE:
            return
        self.chunk_embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.chunk_embedding_cache_dir / f"{key}.pt"
        if path.exists():
            return
        temp_path = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        payload = {
            "cache_version": EMBEDDING_CACHE_VERSION,
            "model": self.model_name,
            "dimensions": OPENAI_EMBEDDING_DIMENSIONS,
            "model_max_length": self.model_max_length,
            "content_tokens_per_chunk": self.content_tokens_per_chunk,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding": embedding.detach().cpu(),
        }
        torch.save(payload, temp_path)
        temp_path.replace(path)

    def load_document_embedding_from_disk(
        self,
        key: str,
    ) -> Optional[DocumentEmbedding]:
        if not ENABLE_EMBEDDING_DISK_CACHE:
            return None
        path = self.document_embedding_cache_dir / f"{key}.pt"
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu")
            document_embedding = payload["document_embedding"].float().cpu()
            chunk_embeddings = payload["chunk_embeddings"].float().cpu()
            chunk_weights = payload["chunk_weights"]
            chunk_texts = payload["chunk_texts"]
        except Exception:
            return None
        if not isinstance(chunk_embeddings, torch.Tensor):
            return None
        if chunk_embeddings.dim() == 1:
            chunk_embedding_list = [chunk_embeddings]
        else:
            chunk_embedding_list = [embedding for embedding in chunk_embeddings]
        if isinstance(chunk_weights, torch.Tensor):
            chunk_weight_list = [int(value) for value in chunk_weights.tolist()]
        else:
            chunk_weight_list = [int(value) for value in chunk_weights]
        return DocumentEmbedding(
            document_embedding=document_embedding,
            chunk_embeddings=chunk_embedding_list,
            chunk_weights=chunk_weight_list,
            chunk_texts=[str(text) for text in chunk_texts],
        )

    def save_document_embedding_to_disk(
        self,
        key: str,
        text: str,
        document_embedding: DocumentEmbedding,
    ) -> None:
        if not ENABLE_EMBEDDING_DISK_CACHE:
            return
        self.document_embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.document_embedding_cache_dir / f"{key}.pt"
        if path.exists():
            return
        temp_path = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        payload = {
            "cache_version": EMBEDDING_CACHE_VERSION,
            "model": self.model_name,
            "dimensions": OPENAI_EMBEDDING_DIMENSIONS,
            "model_max_length": self.model_max_length,
            "content_tokens_per_chunk": self.content_tokens_per_chunk,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "document_embedding": document_embedding.document_embedding.detach().cpu(),
            "chunk_embeddings": torch.stack(
                [embedding.detach().cpu() for embedding in document_embedding.chunk_embeddings]
            ),
            "chunk_weights": torch.tensor(
                document_embedding.chunk_weights,
                dtype=torch.int64,
            ),
            "chunk_texts": document_embedding.chunk_texts,
        }
        torch.save(payload, temp_path)
        temp_path.replace(path)

    def token_ids(self, text: str) -> list[int]:
        cleaned = str(text or "")
        return self.encoding.encode(cleaned, disallowed_special=())

    def split_sentences(self, paragraph: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", paragraph.strip())
        if not normalized:
            return []
        pieces = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", normalized)
        return [piece.strip() for piece in pieces if piece.strip()]

    def split_long_sentence_by_token_window(
        self,
        sentence: str,
        token_ids: list[int],
    ) -> list[tuple[str, int]]:
        chunks: list[tuple[str, int]] = []
        for start in range(0, len(token_ids), self.content_tokens_per_chunk):
            chunk_ids = token_ids[start : start + self.content_tokens_per_chunk]
            chunk_text = self.encoding.decode(chunk_ids).strip()
            if not chunk_text:
                chunk_text = sentence
            chunks.append((chunk_text, len(chunk_ids)))
        return chunks

    def natural_chunks(self, text: str) -> list[tuple[str, int]]:
        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("Cannot chunk an empty text.")
        with self.cache_lock:
            cached = self.chunk_cache.get(cleaned)
        if cached is not None:
            return cached

        chunks: list[tuple[str, int]] = []
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", cleaned)
            if paragraph.strip()
        ]

        for paragraph in paragraphs:
            current_sentences: list[str] = []
            current_token_count = 0

            for sentence in self.split_sentences(paragraph):
                sentence_ids = self.token_ids(sentence)
                sentence_token_count = len(sentence_ids)
                if sentence_token_count == 0:
                    continue

                if sentence_token_count > self.content_tokens_per_chunk:
                    if current_sentences:
                        chunks.append((" ".join(current_sentences), current_token_count))
                        current_sentences = []
                        current_token_count = 0
                    chunks.extend(
                        self.split_long_sentence_by_token_window(
                            sentence,
                            sentence_ids,
                        )
                    )
                    continue

                would_fit = (
                    current_token_count + sentence_token_count
                    <= self.content_tokens_per_chunk
                )
                if current_sentences and not would_fit:
                    chunks.append((" ".join(current_sentences), current_token_count))
                    current_sentences = []
                    current_token_count = 0

                current_sentences.append(sentence)
                current_token_count += sentence_token_count

            if current_sentences:
                chunks.append((" ".join(current_sentences), current_token_count))

        if not chunks:
            token_ids = self.token_ids(cleaned)
            chunks = self.split_long_sentence_by_token_window(cleaned, token_ids)

        with self.cache_lock:
            self.chunk_cache[cleaned] = chunks
        return chunks

    def request_embedding(self, text: str) -> torch.Tensor:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
        }
        if OPENAI_EMBEDDING_DIMENSIONS is not None:
            request_kwargs["dimensions"] = OPENAI_EMBEDDING_DIMENSIONS

        last_error: Optional[Exception] = None
        for attempt in range(OPENAI_EMBEDDING_RETRIES):
            try:
                response = self.client.embeddings.create(**request_kwargs)
                vector = response.data[0].embedding
                embedding = torch.tensor(vector, dtype=torch.float32)
                return torch.nn.functional.normalize(embedding, p=2, dim=0)
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= OPENAI_EMBEDDING_RETRIES:
                    break
                time.sleep(OPENAI_EMBEDDING_RETRY_BASE_SECONDS * (2**attempt))

        raise RuntimeError("OpenAI embedding request failed.") from last_error

    def encode_chunk(self, text: str) -> torch.Tensor:
        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("Cannot embed an empty text.")
        with self.cache_lock:
            cached = self.chunk_embedding_cache.get(cleaned)
        if cached is not None:
            return cached

        cache_key = self.cache_key(cleaned)
        with self.lock_for_key(f"chunk:{cache_key}"):
            with self.cache_lock:
                cached = self.chunk_embedding_cache.get(cleaned)
            if cached is not None:
                return cached

            cached = self.load_chunk_embedding_from_disk(cache_key)
            if cached is not None:
                with self.cache_lock:
                    self.chunk_embedding_cache[cleaned] = cached
                return cached

            input_ids = self.token_ids(cleaned)
            if not input_ids:
                raise ValueError("Tokenizer produced no tokens for non-empty text.")

            encoded_text = cleaned
            if len(input_ids) > self.content_tokens_per_chunk:
                input_ids = input_ids[: self.content_tokens_per_chunk]
                encoded_text = self.encoding.decode(input_ids).strip()
                if not encoded_text:
                    encoded_text = cleaned

            embedding = self.request_embedding(encoded_text)
            with self.cache_lock:
                self.chunk_embedding_cache[cleaned] = embedding
            self.save_chunk_embedding_to_disk(cache_key, cleaned, embedding)
            return embedding

    def encode_document(self, text: str) -> DocumentEmbedding:
        cleaned = str(text or "").strip()
        if not cleaned:
            raise ValueError("Cannot embed an empty document.")
        with self.cache_lock:
            cached = self.document_cache.get(cleaned)
        if cached is not None:
            return cached

        cache_key = self.cache_key(cleaned)
        with self.lock_for_key(f"document:{cache_key}"):
            with self.cache_lock:
                cached = self.document_cache.get(cleaned)
            if cached is not None:
                return cached

            cached = self.load_document_embedding_from_disk(cache_key)
            if cached is not None:
                with self.cache_lock:
                    self.document_cache[cleaned] = cached
                return cached

            chunks = self.natural_chunks(cleaned)
            chunk_texts = [chunk_text for chunk_text, _ in chunks]
            chunk_weights = [max(1, token_count) for _, token_count in chunks]
            chunk_embeddings = [
                self.encode_chunk(chunk_text) for chunk_text in chunk_texts
            ]

            weights = torch.tensor(chunk_weights, dtype=torch.float32).unsqueeze(1)
            stacked = torch.stack(chunk_embeddings)
            embedding = (stacked * weights).sum(dim=0) / weights.sum()
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
            document_embedding = DocumentEmbedding(
                document_embedding=embedding,
                chunk_embeddings=chunk_embeddings,
                chunk_weights=chunk_weights,
                chunk_texts=chunk_texts,
            )
            with self.cache_lock:
                self.document_cache[cleaned] = document_embedding
            self.save_document_embedding_to_disk(
                cache_key,
                cleaned,
                document_embedding,
            )
            return document_embedding

    def similarity(self, left: str, right: str) -> SimilarityScores:
        """Compute the same 0.3-global/0.7-coverage score used by Figure 3."""
        left_document = self.encode_document(left)
        right_document = self.encode_document(right)
        s_global = float(
            torch.dot(
                left_document.document_embedding,
                right_document.document_embedding,
            ).item()
        )
        left_matrix = torch.stack(left_document.chunk_embeddings)
        right_matrix = torch.stack(right_document.chunk_embeddings)
        similarities = left_matrix @ right_matrix.T
        left_weights = torch.tensor(left_document.chunk_weights, dtype=torch.float32)
        right_weights = torch.tensor(right_document.chunk_weights, dtype=torch.float32)
        left_coverage = float(
            (similarities.max(dim=1).values * left_weights).sum()
            / left_weights.sum()
        )
        right_coverage = float(
            (similarities.max(dim=0).values * right_weights).sum()
            / right_weights.sum()
        )
        s_coverage = (left_coverage + right_coverage) / 2.0
        return SimilarityScores(
            s_global=s_global,
            s_coverage=s_coverage,
            s_final=0.3 * s_global + 0.7 * s_coverage,
        )


def resolve_source_models(selection: object) -> list[str]:
    if selection is None:
        selection = CACHE_SOURCE_MODELS
    if isinstance(selection, str):
        requested = [selection]
    else:
        requested = [str(model) for model in selection]

    normalized = list(
        dict.fromkeys(model.strip() for model in requested if str(model).strip())
    )
    if not normalized or any(model.lower() == "all" for model in normalized):
        return list(SELECTED_SOURCE_MODELS)

    known_models = set(SELECTED_SOURCE_MODELS)
    unknown = [model for model in normalized if model not in known_models]
    if unknown:
        print(
            "[WARN] Source model(s) not listed in SELECTED_SOURCE_MODELS; "
            f"will scan if matching output folders exist: {', '.join(unknown)}"
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute text-embedding-3-large cache from the bundled "
            "compressed source table."
        )
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGY_PATHS),
        default=list(STRATEGY_PATHS),
        help="Strategies to scan. Default: all configured strategies.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Source model names to include, or 'all'. "
            f"Default comes from CACHE_SOURCE_MODELS: {CACHE_SOURCE_MODELS!r}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan inputs and report missing cache entries without encoding.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N valid output/reference pairs. Default: 200.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CACHE_WORKERS,
        help=f"Parallel document encoding workers. Default: {CACHE_WORKERS}.",
    )
    parser.add_argument(
        "--source-table",
        type=Path,
        default=DEFAULT_SOURCE_TABLE,
        help="Bundled compressed source table. No raw TXT/DB directories are read.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=EMBEDDING_CACHE_DIR,
        help="Local cache root. Default: embedding_cache beside this script.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Optional source-row limit for a small smoke test.",
    )
    return parser.parse_args()


def queue_document_cache_job(
    embedder,
    text: str,
    seen_cache_keys: set[str],
    encoding_jobs: list[EncodingJob],
    stats: CacheStats,
    *,
    dry_run: bool,
    label: str,
) -> None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return

    cache_key = embedder.cache_key(cleaned)
    if cache_key in seen_cache_keys:
        stats.skipped_duplicate_this_run += 1
        return
    seen_cache_keys.add(cache_key)

    cache_path = embedder.document_embedding_cache_dir / f"{cache_key}.pt"
    if cache_path.exists():
        stats.skipped_existing_cache += 1
        return

    if dry_run:
        stats.would_encode += 1
        return

    encoding_jobs.append(EncodingJob(text=cleaned, label=label))


def encode_missing_documents(
    embedder,
    encoding_jobs: list[EncodingJob],
    *,
    workers: int,
    progress_every: int,
) -> tuple[int, int]:
    if not encoding_jobs:
        return 0, 0

    total = len(encoding_jobs)
    worker_count = max(1, min(workers, total))
    print(f"\nEncoding {total} missing document cache file(s) with {worker_count} worker(s)")

    completed = 0
    encoded = 0
    errors = 0
    progress_step = progress_every if progress_every > 0 else max(1, total // 10)

    if worker_count == 1:
        for job in encoding_jobs:
            try:
                embedder.encode_document(job.text)
                encoded += 1
            except Exception as exc:
                errors += 1
                print(f"[ERROR] Failed to encode {job.label}: {exc}")
            completed += 1
            if completed == total or completed % progress_step == 0:
                print(f"  encoded {completed}/{total}")
        return encoded, errors

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(embedder.encode_document, job.text): job
            for job in encoding_jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                encoded += 1
            except Exception as exc:
                errors += 1
                print(f"[ERROR] Failed to encode {job.label}: {exc}")
            completed += 1
            if completed == total or completed % progress_step == 0:
                print(f"  encoded {completed}/{total}")

    return encoded, errors


def print_stats(stats: CacheStats, *, dry_run: bool) -> None:
    action_label = "Would encode" if dry_run else "Newly encoded"
    print("\nDone.")
    print(f"  Source records seen: {stats.output_txt_seen}")
    print(f"  Valid output/reference pairs: {stats.valid_pairs}")
    print(f"  Skipped unmatched records: {stats.skipped_unmatched_output}")
    print(
        "  Skipped empty mechanistic_explanation: "
        f"{stats.skipped_empty_mechanistic_explanation}"
    )
    print(f"  Skipped empty output text: {stats.skipped_empty_output_txt}")
    print(f"  Skipped duplicate text this run: {stats.skipped_duplicate_this_run}")
    print(f"  Skipped existing document cache: {stats.skipped_existing_cache}")
    print(f"  {action_label}: {stats.would_encode if dry_run else stats.newly_encoded}")
    print(f"  Errors: {stats.errors}")


def main() -> None:
    global EMBEDDING_CACHE_DIR
    args = parse_args()
    EMBEDDING_CACHE_DIR = args.cache_dir.resolve()
    selected_models = resolve_source_models(args.models)
    strategy_codes = {strategy_label_to_code(name) for name in args.strategies}
    if args.limit_records is not None and args.limit_records < 1:
        raise SystemExit("--limit-records must be positive")

    print(f"Source table: {args.source_table.resolve()}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"OpenAI base URL: {OPENAI_BASE_URL or 'SDK default'}")
    print(f"Source models: {', '.join(selected_models)}")
    print(f"Strategies: {', '.join(sorted(strategy_codes))}")
    print(f"Cache workers: {args.workers}")

    if args.dry_run:
        embedder = CacheKeyOnlyEmbedder(EMBEDDING_MODEL)
    else:
        embedder = OpenAIEmbedder(EMBEDDING_MODEL)
    print(f"Embedding cache dir: {embedder.cache_dir}")

    stats = CacheStats()
    seen_cache_keys: set[str] = set()
    encoding_jobs: list[EncodingJob] = []
    for record in iter_source_records(
        args.source_table,
        source_models=set(selected_models),
        strategies=strategy_codes,
        limit=args.limit_records,
    ):
        stats.output_txt_seen += 1
        if not record.mechanistic_explanation or not record.reference_text:
            stats.skipped_empty_mechanistic_explanation += 1
            continue
        if not record.output_text:
            stats.skipped_empty_output_txt += 1
            continue
        stats.valid_pairs += 1
        label = (
            f"{record.strategy} | {record.source_model} | Round {record.round_number} "
            f"| molecule {record.molecule_id} experiment {record.experiment_id}"
        )
        queue_document_cache_job(
            embedder, record.reference_text, seen_cache_keys, encoding_jobs, stats,
            dry_run=args.dry_run, label=f"{label} reference",
        )
        queue_document_cache_job(
            embedder, record.output_text, seen_cache_keys, encoding_jobs, stats,
            dry_run=args.dry_run, label=f"{label} output",
        )
        if args.progress_every > 0 and stats.valid_pairs % args.progress_every == 0:
            queued = stats.would_encode if args.dry_run else len(encoding_jobs)
            print(f"  valid pairs: {stats.valid_pairs}; queued: {queued}")

    if not args.dry_run:
        encoded, errors = encode_missing_documents(
            embedder,
            encoding_jobs,
            workers=args.workers,
            progress_every=args.progress_every,
        )
        stats.newly_encoded += encoded
        stats.errors += errors

    print_stats(stats, dry_run=args.dry_run)

    if stats.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


