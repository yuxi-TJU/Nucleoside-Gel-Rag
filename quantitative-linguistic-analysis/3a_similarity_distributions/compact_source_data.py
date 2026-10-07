"""Read the bundled compressed source table without repository-external files."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_TABLE = (
    SCRIPT_DIR
    / "source_data"
    / "max0_and_max3_descriptor_morgan_source_records.csv.gz"
)


@dataclass(frozen=True)
class CompactSourceRecord:
    source_model: str
    retrieval_family: str
    strategy: str
    max_k: int
    round_number: int
    molecule_id: int
    experiment_id: int
    chemical_description: str
    solvent_additive: str
    expected_result: str
    mechanistic_explanation: str
    reference_text: str
    output_text: str
    prediction_json: str


def strategy_label_to_code(label: str) -> str:
    normalized = str(label).strip()
    if normalized.startswith("Strategy "):
        return "S" + normalized.removeprefix("Strategy ").strip()
    if normalized.startswith("S"):
        return normalized
    raise ValueError(f"Unknown strategy label: {label!r}")


def iter_source_records(
    path: Path = DEFAULT_SOURCE_TABLE,
    *,
    source_models: set[str] | None = None,
    strategies: set[str] | None = None,
    limit: int | None = None,
) -> Iterator[CompactSourceRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Compressed source table not found: {path}\n"
            "Keep source_data/ with this folder, or pass --source-table."
        )
    yielded = 0
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_model", "retrieval_family", "strategy", "max_k", "round",
            "molecule_id", "experiment_id", "chemical_description",
            "solvent_additive", "expected_result", "mechanistic_explanation",
            "reference_text", "output_text", "prediction_json",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Source table {path} is missing columns: {', '.join(missing)}"
            )
        for row in reader:
            model = str(row["source_model"]).strip()
            strategy = str(row["strategy"]).strip()
            if source_models is not None and model not in source_models:
                continue
            if strategies is not None and strategy not in strategies:
                continue
            yield CompactSourceRecord(
                source_model=model,
                retrieval_family=str(row["retrieval_family"]).strip(),
                strategy=strategy,
                max_k=int(row["max_k"]),
                round_number=int(row["round"]),
                molecule_id=int(row["molecule_id"]),
                experiment_id=int(row["experiment_id"]),
                chemical_description=str(row["chemical_description"]),
                solvent_additive=str(row["solvent_additive"]),
                expected_result=str(row["expected_result"]).strip(),
                mechanistic_explanation=str(row["mechanistic_explanation"]).strip(),
                reference_text=str(row["reference_text"]).strip(),
                output_text=str(row["output_text"]).strip(),
                prediction_json=str(row["prediction_json"]),
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def predicted_result(prediction_json: str) -> str | None:
    """Return a unique Gel/No Gel result from one stored prediction payload."""
    try:
        payload = json.loads(prediction_json)
    except json.JSONDecodeError:
        return None
    results: list[str] = []
    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        if not isinstance(item, dict):
            continue
        experiments = item.get("experiments", [])
        if not isinstance(experiments, list):
            continue
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            value = str(experiment.get("results", "")).strip().lower()
            if value == "gel":
                results.append("Gel")
            elif value in {"no gel", "no-gel", "nogel"}:
                results.append("No Gel")
    unique = set(results)
    return next(iter(unique)) if len(unique) == 1 else None


def normalize_result(value: str) -> str | None:
    normalized = str(value).strip().lower().replace("-", " ")
    normalized = " ".join(normalized.split())
    if normalized == "gel":
        return "Gel"
    if normalized in {"no gel", "nogel"}:
        return "No Gel"
    return None
