#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BUNDLE_DIR / "molecule_database.json"
DEFAULT_DESCRIPTOR_VALUES_PATH = BUNDLE_DIR / "molecule_descriptor_values_24d.csv"
DEFAULT_OUTPUT_DIR = BUNDLE_DIR
DEFAULT_MAX_K_VALUES = (3, 5, 6)
DEFAULT_TANIMOTO_RADIUS = 2
DEFAULT_TANIMOTO_BITS = 2048
REFERENCE_STRATEGIES = (
    ("S1", "S1_RAG_descriptors_masked.py", "descriptor"),
    ("S2", "S2_RAG_descriptors_conditions_masked.py", "condition_aware"),
    ("S3", "S3_RAG_descriptors_conditions.py", "condition_aware"),
    ("S1_1", "S1_1_RAG_tanimoto_masked.py", "tanimoto"),
    ("S2_1", "S2_1_RAG_tanimoto_conditions_masked.py", "tanimoto_condition_aware"),
    ("S3_1", "S3_1_RAG_tanimoto_conditions.py", "tanimoto_condition_aware"),
)
PAIRWISE_COMPARISONS = tuple(
    (strategy_a[0], strategy_b[0])
    for strategy_a, strategy_b in combinations(REFERENCE_STRATEGIES, 2)
)

SEPARATOR_PATTERN = re.compile(r"\s*(?:;|\+|\bor\b|\band\b)\s*", flags=re.IGNORECASE)
DESCRIPTOR_METADATA_COLUMNS = {"molecule_index", "chemical_description", "canonical_smiles"}
EXPECTED_DESCRIPTOR_COUNT = 24


@dataclass(frozen=True)
class SingleConditionTarget:
    molecule_idx: int
    molecule: dict[str, Any]
    experiment: dict[str, Any]
    experiment_number: int

    @property
    def molecule_index(self) -> str:
        return str(self.molecule["molecule_index"])

    @property
    def chemical_description(self) -> str:
        return str(self.molecule["chemical_description"])

    @property
    def solvent_additive(self) -> str:
        return str(self.experiment.get("solvent_additive", "")).strip()

    @property
    def detailed_conditions(self) -> str:
        return str(self.experiment.get("detailed_conditions", "")).strip()


@dataclass(frozen=True)
class ReferenceSetRecord:
    molecule_index: str
    chemical_description: str
    experiment_number: int
    solvent_additive: str
    detailed_conditions: str
    strategy: str
    strategy_script: str
    max_k: int
    candidate_molecule_count: int
    selected_reference_indices: tuple[str, ...]
    selected_reference_descriptions: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseOverlapRecord:
    comparison_type: str
    comparison_columns: str
    molecule_index: str
    chemical_description: str
    experiment_number: int
    solvent_additive: str
    detailed_conditions: str
    max_k: int
    strategy_a: str
    strategy_b: str
    intersection_size: int
    union_size: int
    jaccard_overlap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute molecule-level reference overlap for S1, S2, S3, S1_1, S2_1, and S3_1 retrieval."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--descriptor-values",
        type=Path,
        default=DEFAULT_DESCRIPTOR_VALUES_PATH,
        help="Descriptor matrix CSV; descriptor columns are read from its header.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_K_VALUES),
        help="One or more retrieval depths to evaluate, e.g. --max-k 3 5 6.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument(
        "--experiment-indices",
        type=int,
        nargs="*",
        help="Optional 1-based experiment numbers to evaluate within each selected molecule.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_descriptor_matrix_from_csv(
    path: Path,
    db: list[dict[str, Any]],
) -> Any:

    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)
    if "molecule_index" not in df.columns:
        raise KeyError("Descriptor value CSV is missing the molecule_index column.")
    descriptor_names = [
        column for column in df.columns if column not in DESCRIPTOR_METADATA_COLUMNS
    ]
    if len(descriptor_names) != EXPECTED_DESCRIPTOR_COUNT:
        raise ValueError(
            "Descriptor value CSV must contain exactly "
            f"{EXPECTED_DESCRIPTOR_COUNT} descriptor columns; found {len(descriptor_names)}."
        )
    df[descriptor_names] = df[descriptor_names].apply(pd.to_numeric, errors="coerce")
    if df[descriptor_names].isna().any().any():
        invalid_columns = [column for column in descriptor_names if df[column].isna().any()]
        raise ValueError(f"Invalid or missing descriptor values in columns: {invalid_columns}")

    df["molecule_index"] = df["molecule_index"].astype(str)
    df = df.set_index("molecule_index", drop=False)
    rows = []
    missing_indices: list[str] = []
    for item in db:
        molecule_index = str(item["molecule_index"])
        if molecule_index not in df.index:
            missing_indices.append(molecule_index)
            continue
        rows.append(df.loc[molecule_index, descriptor_names].to_numpy(dtype=float))
    if missing_indices:
        raise KeyError(f"Descriptor value CSV is missing molecule_index values: {missing_indices}")
    return np.asarray(rows, dtype=np.float64)


def compute_distance_matrix_from_values(descriptor_matrix: Any) -> Any:
    import numpy as np

    values = np.asarray(descriptor_matrix, dtype=np.float64)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    scaled = (values - means) / stds
    distance_matrix = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=2)
    np.fill_diagonal(distance_matrix, np.inf)
    return distance_matrix


def compute_distance_matrix(
    db: list[dict[str, Any]],
    descriptor_values_path: Path,
) -> Any:
    descriptor_matrix = load_descriptor_matrix_from_csv(descriptor_values_path, db)
    return compute_distance_matrix_from_values(descriptor_matrix)


def compute_tanimoto_distance_matrix(db: list[dict[str, Any]]) -> Any:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    import numpy as np

    fingerprints = []
    for item in db:
        smiles = str(item["canonical_smiles"]).strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(
                "RDKit could not parse canonical_smiles for molecule_index "
                f"{item.get('molecule_index')}: {smiles}"
            )
        fingerprints.append(
            AllChem.GetMorganFingerprintAsBitVect(
                mol,
                DEFAULT_TANIMOTO_RADIUS,
                nBits=DEFAULT_TANIMOTO_BITS,
            )
        )

    size = len(fingerprints)
    distance_matrix = np.full((size, size), np.inf, dtype=np.float64)
    for idx, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
        distance_matrix[idx, :] = [1.0 - float(similarity) for similarity in similarities]
    np.fill_diagonal(distance_matrix, np.inf)
    return distance_matrix


def normalize_solvent_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "").strip())
    normalized = normalized.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+\([^)]*\)$", "", normalized)
    return normalized


def canonicalize_solvent_token(token: str) -> str:
    token = normalize_solvent_text(token).lower()
    token = re.sub(r"\s+", "", token)
    token = token.strip(",.:")
    if token in {"b(oh)3", "h3bo3"}:
        return "boricacid"
    return token


def split_solvent_keywords(solvent_additive: Any) -> frozenset[str]:
    text = normalize_solvent_text(solvent_additive)
    if not text:
        return frozenset()
    parts = [part.strip() for part in SEPARATOR_PATTERN.split(text) if part.strip()]
    if not parts:
        parts = [text]
    tokens = {canonicalize_solvent_token(part) for part in parts}
    tokens.discard("")
    return frozenset(tokens)


def filter_molecules(db: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.indices:
        wanted = {int(index) for index in args.indices}
        return [
            (idx, item)
            for idx, item in enumerate(db)
            if int(item["molecule_index"]) in wanted
        ]

    selected = []
    for idx, item in enumerate(db):
        molecule_index = int(item["molecule_index"])
        if molecule_index < args.start_index:
            continue
        if args.end_index is not None and molecule_index > args.end_index:
            continue
        selected.append((idx, item))
    return selected


def iter_single_condition_targets(
    db: list[dict[str, Any]],
    molecule_items: list[tuple[int, dict[str, Any]]],
    experiment_indices: Iterable[int] | None,
) -> Iterable[SingleConditionTarget]:
    experiment_filter = set(experiment_indices or [])
    for molecule_idx, molecule in molecule_items:
        for experiment_number, experiment in enumerate(molecule.get("experiments", []), start=1):
            if experiment_filter and experiment_number not in experiment_filter:
                continue
            if not str(experiment.get("solvent_additive", "")).strip():
                continue
            yield SingleConditionTarget(
                molecule_idx=molecule_idx,
                molecule=molecule,
                experiment=experiment,
                experiment_number=experiment_number,
            )


def ranked_distance_references(
    db: list[dict[str, Any]],
    distance_matrix: Any,
    target_idx: int,
    max_k: int,
    excluded_indices: Iterable[int] | None = None,
) -> tuple[int, ...]:
    excluded = set(excluded_indices or ())
    excluded.add(target_idx)
    ranked_indices = sorted(
        (idx for idx in range(len(db)) if idx not in excluded),
        key=lambda idx: (float(distance_matrix[target_idx, idx]), int(db[idx]["molecule_index"])),
    )
    return tuple(ranked_indices[:max_k])


def molecule_matches_condition(molecule: dict[str, Any], target_tokens: frozenset[str]) -> bool:
    if not target_tokens:
        return False
    for exp in molecule.get("experiments", []):
        exp_tokens = split_solvent_keywords(exp.get("solvent_additive", ""))
        if exp_tokens & target_tokens:
            return True
    return False


def ranked_condition_aware_references(
    db: list[dict[str, Any]],
    distance_matrix: Any,
    target: SingleConditionTarget,
    max_k: int,
) -> tuple[tuple[int, ...], int]:
    target_tokens = split_solvent_keywords(target.solvent_additive)
    candidates = [
        idx
        for idx, molecule in enumerate(db)
        if idx != target.molecule_idx and molecule_matches_condition(molecule, target_tokens)
    ]
    candidates.sort(
        key=lambda idx: (
            float(distance_matrix[target.molecule_idx, idx]),
            int(db[idx]["molecule_index"]),
        )
    )
    if not candidates:
        fallback = ranked_distance_references(db, distance_matrix, target.molecule_idx, max_k)
        return fallback, 0

    selected = list(candidates[:max_k])
    if len(selected) < max_k:
        selected_set = set(selected)
        selected.extend(
            ranked_distance_references(
                db=db,
                distance_matrix=distance_matrix,
                target_idx=target.molecule_idx,
                max_k=max_k - len(selected),
                excluded_indices=selected_set,
            )
        )

    return tuple(selected), len(candidates)


def build_reference_ids(
    db: list[dict[str, Any]],
    reference_indices: Iterable[int],
) -> tuple[str, ...]:
    return tuple(str(db[idx]["molecule_index"]) for idx in reference_indices)


def shorten_text(value: Any, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def build_reference_descriptions(
    db: list[dict[str, Any]],
    reference_indices: Iterable[int],
) -> tuple[str, ...]:
    return tuple(
        f"{db[idx]['molecule_index']} | {shorten_text(db[idx].get('chemical_description', ''))}"
        for idx in reference_indices
    )


def make_reference_set_record(
    target: SingleConditionTarget,
    strategy: str,
    strategy_script: str,
    max_k: int,
    candidate_molecule_count: int,
    reference_indices: tuple[int, ...],
    db: list[dict[str, Any]],
) -> ReferenceSetRecord:
    return ReferenceSetRecord(
        molecule_index=target.molecule_index,
        chemical_description=target.chemical_description,
        experiment_number=target.experiment_number,
        solvent_additive=target.solvent_additive,
        detailed_conditions=target.detailed_conditions,
        strategy=strategy,
        strategy_script=strategy_script,
        max_k=max_k,
        candidate_molecule_count=candidate_molecule_count,
        selected_reference_indices=build_reference_ids(db, reference_indices),
        selected_reference_descriptions=build_reference_descriptions(db, reference_indices),
    )


def build_reference_set_records(
    db: list[dict[str, Any]],
    descriptor_distance_matrix: Any,
    tanimoto_distance_matrix: Any,
    targets: list[SingleConditionTarget],
    max_k_values: list[int],
) -> list[ReferenceSetRecord]:
    records: list[ReferenceSetRecord] = []
    for target in targets:
        for max_k in max_k_values:
            descriptor_refs = ranked_distance_references(
                db,
                descriptor_distance_matrix,
                target.molecule_idx,
                max_k,
            )
            tanimoto_refs = ranked_distance_references(
                db,
                tanimoto_distance_matrix,
                target.molecule_idx,
                max_k,
            )
            condition_refs, condition_candidate_count = ranked_condition_aware_references(
                db,
                descriptor_distance_matrix,
                target,
                max_k,
            )
            tanimoto_condition_refs, tanimoto_condition_candidate_count = ranked_condition_aware_references(
                db,
                tanimoto_distance_matrix,
                target,
                max_k,
            )

            for strategy, strategy_script, strategy_type in REFERENCE_STRATEGIES:
                if strategy_type == "descriptor":
                    reference_indices = descriptor_refs
                    candidate_molecule_count = len(db) - 1
                else:
                    if strategy_type == "condition_aware":
                        reference_indices = condition_refs
                        candidate_molecule_count = condition_candidate_count
                    elif strategy_type == "tanimoto":
                        reference_indices = tanimoto_refs
                        candidate_molecule_count = len(db) - 1
                    elif strategy_type == "tanimoto_condition_aware":
                        reference_indices = tanimoto_condition_refs
                        candidate_molecule_count = tanimoto_condition_candidate_count
                    else:
                        raise ValueError(f"Unknown strategy type: {strategy_type}")

                records.append(
                    make_reference_set_record(
                        target=target,
                        strategy=strategy,
                        strategy_script=strategy_script,
                        max_k=max_k,
                        candidate_molecule_count=candidate_molecule_count,
                        reference_indices=reference_indices,
                        db=db,
                    )
                )
    return records


def jaccard_overlap(a: Iterable[str], b: Iterable[str]) -> tuple[int, int, float]:
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    intersection = set_a & set_b
    if not union:
        return 0, 0, math.nan
    return len(intersection), len(union), len(intersection) / len(union)


def build_pairwise_records(records: list[ReferenceSetRecord]) -> list[PairwiseOverlapRecord]:
    by_key: dict[tuple[str, int, int], dict[str, ReferenceSetRecord]] = {}
    for record in records:
        key = (record.molecule_index, record.experiment_number, record.max_k)
        by_key.setdefault(key, {})[record.strategy] = record

    pairwise_records: list[PairwiseOverlapRecord] = []
    for strategy_records in by_key.values():
        for strategy_a, strategy_b in PAIRWISE_COMPARISONS:
            record_a = strategy_records.get(strategy_a)
            record_b = strategy_records.get(strategy_b)
            if not record_a or not record_b:
                continue
            intersection_size, union_size, score = jaccard_overlap(
                record_a.selected_reference_indices,
                record_b.selected_reference_indices,
            )
            pairwise_records.append(
                PairwiseOverlapRecord(
                    comparison_type=f"{strategy_a}_vs_{strategy_b}",
                    comparison_columns=(
                        f"{strategy_a} {strategy_label_from_script(record_a.strategy_script)} vs "
                        f"{strategy_b} {strategy_label_from_script(record_b.strategy_script)}"
                    ),
                    molecule_index=record_a.molecule_index,
                    chemical_description=record_a.chemical_description,
                    experiment_number=record_a.experiment_number,
                    solvent_additive=record_a.solvent_additive,
                    detailed_conditions=record_a.detailed_conditions,
                    max_k=record_a.max_k,
                    strategy_a=record_a.strategy,
                    strategy_b=record_b.strategy,
                    intersection_size=intersection_size,
                    union_size=union_size,
                    jaccard_overlap=score,
                )
            )
    return pairwise_records


def strategy_label_from_script(script_name: str) -> str:
    stem = Path(script_name).stem
    return re.sub(r"^S\d+(?:_\d+)?_RAG_", "", stem)


def is_perfect_overlap(record: PairwiseOverlapRecord) -> bool:
    return not math.isnan(record.jaccard_overlap) and math.isclose(
        record.jaccard_overlap,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def filter_display_pairwise_records(
    records: list[PairwiseOverlapRecord],
) -> list[PairwiseOverlapRecord]:
    return [record for record in records if not is_perfect_overlap(record)]


def write_reference_sets(path: Path, records: list[ReferenceSetRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "molecule_index",
            "chemical_description",
            "experiment_number",
            "solvent_additive",
            "detailed_conditions",
            "strategy",
            "strategy_script",
            "max_k",
            "candidate_molecule_count",
            "selected_reference_count",
            "selected_reference_indices",
            "selected_reference_descriptions",
        ])
        for record in records:
            writer.writerow([
                record.molecule_index,
                record.chemical_description,
                record.experiment_number,
                record.solvent_additive,
                record.detailed_conditions,
                record.strategy,
                record.strategy_script,
                record.max_k,
                record.candidate_molecule_count,
                len(record.selected_reference_indices),
                "; ".join(record.selected_reference_indices),
                "; ".join(record.selected_reference_descriptions),
            ])


def write_pairwise(path: Path, records: list[PairwiseOverlapRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "comparison_type",
            "comparison_columns",
            "molecule_index",
            "chemical_description",
            "experiment_number",
            "solvent_additive",
            "detailed_conditions",
            "max_k",
            "strategy_a",
            "strategy_b",
            "intersection_size",
            "union_size",
            "jaccard_overlap",
        ])
        for record in records:
            writer.writerow([
                record.comparison_type,
                record.comparison_columns,
                record.molecule_index,
                record.chemical_description,
                record.experiment_number,
                record.solvent_additive,
                record.detailed_conditions,
                record.max_k,
                record.strategy_a,
                record.strategy_b,
                record.intersection_size,
                record.union_size,
                f"{record.jaccard_overlap:.6f}",
            ])


def write_summary(path: Path, records: list[PairwiseOverlapRecord]) -> None:
    groups: dict[tuple[str, str, int], list[float]] = {}
    for record in records:
        if math.isnan(record.jaccard_overlap):
            continue
        groups.setdefault(
            (record.comparison_type, record.comparison_columns, record.max_k),
            [],
        ).append(record.jaccard_overlap)

    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "comparison_type",
            "comparison_columns",
            "max_k",
            "mean_jaccard",
        ])
        for (comparison_type, comparison_columns, max_k), values in sorted(
            groups.items(),
            key=lambda item: (item[0][2], item[0][0]),
        ):
            writer.writerow([
                comparison_type,
                comparison_columns,
                max_k,
                f"{sum(values) / len(values):.6f}",
            ])


def main() -> None:
    args = parse_args()
    ensure_exists(args.db, "DB file")
    ensure_exists(args.descriptor_values, "Descriptor value CSV")
    max_k_values = sorted({int(value) for value in args.max_k})
    if not max_k_values or any(value <= 0 for value in max_k_values):
        raise ValueError("--max-k values must be positive integers.")

    db = sorted(read_json(args.db), key=lambda item: int(item["molecule_index"]))
    if not isinstance(db, list):
        raise TypeError("DB.json must be a list of molecules.")
    if max(max_k_values) >= len(db):
        raise ValueError(f"Maximum K ({max(max_k_values)}) must be smaller than DB size ({len(db)}).")

    molecule_items = filter_molecules(db, args)
    targets = list(iter_single_condition_targets(db, molecule_items, args.experiment_indices))
    if not targets:
        raise ValueError("No single-condition targets selected.")

    descriptor_distance_matrix = compute_distance_matrix(db, args.descriptor_values)
    tanimoto_distance_matrix = compute_tanimoto_distance_matrix(db)
    reference_records = build_reference_set_records(
        db=db,
        descriptor_distance_matrix=descriptor_distance_matrix,
        tanimoto_distance_matrix=tanimoto_distance_matrix,
        targets=targets,
        max_k_values=max_k_values,
    )
    pairwise_records = build_pairwise_records(reference_records)
    display_pairwise_records = filter_display_pairwise_records(pairwise_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stale_s1_s2_file = args.output_dir / "s1_s2_vs_s3_reference_overlap.csv"
    if stale_s1_s2_file.exists():
        stale_s1_s2_file.unlink()
    write_reference_sets(args.output_dir / "reference_molecule_sets_detailed.csv", reference_records)
    write_pairwise(args.output_dir / "reference_overlap_pairwise.csv", display_pairwise_records)
    write_summary(args.output_dir / "reference_overlap_summary_by_k.csv", display_pairwise_records)

    print(f"Wrote {len(reference_records)} reference-set rows.")
    print(
        f"Wrote {len(display_pairwise_records)} pairwise molecule-overlap rows "
        f"(hidden perfect-overlap rows: {len(pairwise_records) - len(display_pairwise_records)})."
    )
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
