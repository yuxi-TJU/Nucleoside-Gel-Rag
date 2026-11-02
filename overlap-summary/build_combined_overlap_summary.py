#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BUNDLE_DIR = Path(__file__).resolve().parent
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

import build_reference_overlap_summary as reference_overlap

DEFAULT_DB_PATH = BUNDLE_DIR / "molecule_database.json"
DEFAULT_DESCRIPTOR_VALUES_PATH = BUNDLE_DIR / "molecule_descriptor_values_24d.csv"
DEFAULT_LITERATURE_DOI_PATH = BUNDLE_DIR / "literature_doi_by_molecule.csv"
DEFAULT_OUTPUT_DIR = BUNDLE_DIR
DEFAULT_MAX_K_VALUES = (3, 5, 6)
UNIQUE_LITERATURE_FROM_INDEX = 65
DOI_SPLIT_PATTERN = re.compile(r"\s*;\s*")
STRATEGY_SCRIPT_BY_NAME = {
    strategy: strategy_script
    for strategy, strategy_script, _strategy_type in reference_overlap.REFERENCE_STRATEGIES
}


@dataclass
class LiteratureStats:
    strategy: str
    strategy_script: str
    max_k: int
    reference_slots: int = 0
    same_literature_reference_slots: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one total CSV containing pairwise molecule overlap among "
            "S1/S2/S3/S1_1/S2_1/S3_1 and same-literature ratios for each strategy."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--descriptor-values", type=Path, default=DEFAULT_DESCRIPTOR_VALUES_PATH)
    parser.add_argument("--literature-doi", type=Path, default=DEFAULT_LITERATURE_DOI_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional explicit CSV path. Defaults to output-dir/combined_overlap_summary.csv.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_K_VALUES),
        help="One or more retrieval depths. Default: 3 5 6.",
    )
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument(
        "--experiment-indices",
        type=int,
        nargs="*",
        help="Optional 1-based experiment numbers to include.",
    )
    return parser.parse_args()


def normalize_doi(raw_doi: str) -> str:
    doi = unicodedata.normalize("NFKC", str(raw_doi or "")).strip().lower()
    doi = doi.rstrip(".")
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi.strip()


def parse_doi_set(raw_value: str) -> set[str]:
    return {
        normalized
        for part in DOI_SPLIT_PATTERN.split(str(raw_value or ""))
        if (normalized := normalize_doi(part))
    }


def load_literature_sets(path: Path) -> dict[int, set[str]]:
    literature_by_index: dict[int, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            molecule_index = int(row["molecule_index"])
            if molecule_index >= UNIQUE_LITERATURE_FROM_INDEX:
                literature_by_index[molecule_index] = {
                    f"unique_literature_for_molecule_{molecule_index}"
                }
            else:
                literature_by_index[molecule_index] = parse_doi_set(row.get("doi", ""))
    return literature_by_index


def same_literature(
    target_molecule_index: int,
    reference_molecule_index: int,
    literature_by_index: dict[int, set[str]],
) -> bool:
    target_literature = literature_by_index.get(target_molecule_index, set())
    reference_literature = literature_by_index.get(reference_molecule_index, set())
    return bool(target_literature and reference_literature and target_literature & reference_literature)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def format_float(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def strategy_explanation(strategy: str) -> str:
    script_name = STRATEGY_SCRIPT_BY_NAME.get(strategy)
    if not script_name:
        return strategy
    return f"{strategy} {reference_overlap.strategy_label_from_script(script_name)}"


def build_literature_stats(
    reference_records: list[reference_overlap.ReferenceSetRecord],
    literature_by_index: dict[int, set[str]],
) -> list[LiteratureStats]:
    stats_by_key: dict[tuple[str, int], LiteratureStats] = {}

    for record in reference_records:
        key = (record.strategy, record.max_k)
        stats = stats_by_key.setdefault(
            key,
            LiteratureStats(
                strategy=record.strategy,
                strategy_script=record.strategy_script,
                max_k=record.max_k,
            ),
        )
        target_molecule_index = int(record.molecule_index)

        stats.reference_slots += len(record.selected_reference_indices)

        same_count = 0
        for reference_id in record.selected_reference_indices:
            reference_molecule_index = int(reference_id)
            if same_literature(target_molecule_index, reference_molecule_index, literature_by_index):
                same_count += 1

        stats.same_literature_reference_slots += same_count

    return sorted(stats_by_key.values(), key=lambda item: (item.max_k, item.strategy))


def build_total_rows(
    pairwise_records: list[reference_overlap.PairwiseOverlapRecord],
    literature_stats: list[LiteratureStats],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    pairwise_groups: dict[
        tuple[str, str, int, str, str],
        list[reference_overlap.PairwiseOverlapRecord],
    ] = {}

    for record in pairwise_records:
        pairwise_groups.setdefault(
            (
                record.comparison_type,
                record.comparison_columns,
                record.max_k,
                record.strategy_a,
                record.strategy_b,
            ),
            [],
        ).append(record)

    for (
        comparison_type,
        comparison_columns,
        max_k,
        strategy_a,
        strategy_b,
    ), records in sorted(pairwise_groups.items(), key=lambda item: (item[0][2], item[0][0])):
        jaccard_values = [
            record.jaccard_overlap
            for record in records
            if not math.isnan(record.jaccard_overlap)
        ]
        intersection_values = [record.intersection_size for record in records]
        union_values = [record.union_size for record in records]
        if jaccard_values and all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in jaccard_values
        ):
            continue
        rows.append(
            {
                "metric_type": "pairwise_molecule_overlap",
                "max_k": max_k,
                "strategy_explanation": "",
                "comparison_type": comparison_type,
                "comparison_columns": comparison_columns,
                "mean_jaccard_overlap": format_float(mean(jaccard_values)),
                "mean_intersection_size": format_float(mean(intersection_values)),
                "mean_union_size": format_float(mean(union_values)),
                "global_same_literature_reference_ratio": "",
            }
        )

    for stats in literature_stats:
        global_ratio = (
            stats.same_literature_reference_slots / stats.reference_slots
            if stats.reference_slots
            else math.nan
        )
        rows.append(
            {
                "metric_type": "strategy_same_literature_ratio",
                "max_k": stats.max_k,
                "strategy_explanation": strategy_explanation(stats.strategy),
                "comparison_type": "",
                "comparison_columns": "",
                "mean_jaccard_overlap": "",
                "mean_intersection_size": "",
                "mean_union_size": "",
                "global_same_literature_reference_ratio": format_float(global_ratio),
            }
        )

    return rows


def count_pairwise_groups(
    pairwise_records: list[reference_overlap.PairwiseOverlapRecord],
) -> int:
    return len(
        {
            (
                record.comparison_type,
                record.comparison_columns,
                record.max_k,
                record.strategy_a,
                record.strategy_b,
            )
            for record in pairwise_records
        }
    )


def write_total_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "metric_type",
        "max_k",
        "strategy_explanation",
        "comparison_type",
        "comparison_columns",
        "mean_jaccard_overlap",
        "mean_intersection_size",
        "mean_union_size",
        "global_same_literature_reference_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sync_code_to_output_dir(output_dir: Path) -> Path:
    source_path = Path(__file__).resolve()
    target_path = output_dir / source_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_path != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def main() -> None:
    args = parse_args()
    reference_overlap.ensure_exists(args.db, "DB file")
    reference_overlap.ensure_exists(args.descriptor_values, "Descriptor value CSV")
    reference_overlap.ensure_exists(args.literature_doi, "Literature DOI CSV")

    max_k_values = sorted({int(value) for value in args.max_k})
    if not max_k_values or any(value <= 0 for value in max_k_values):
        raise ValueError("--max-k values must be positive integers.")

    db = sorted(reference_overlap.read_json(args.db), key=lambda item: int(item["molecule_index"]))
    if not isinstance(db, list):
        raise TypeError("DB.json must be a list of molecule records.")
    if max(max_k_values) >= len(db):
        raise ValueError(f"Maximum K ({max(max_k_values)}) must be smaller than DB size ({len(db)}).")

    molecule_items = reference_overlap.filter_molecules(db, args)
    targets = list(
        reference_overlap.iter_single_condition_targets(
            db,
            molecule_items,
            args.experiment_indices,
        )
    )
    if not targets:
        raise ValueError("No single-condition targets selected.")

    descriptor_distance_matrix = reference_overlap.compute_distance_matrix(
        db,
        args.descriptor_values,
    )
    tanimoto_distance_matrix = reference_overlap.compute_tanimoto_distance_matrix(db)
    reference_records = reference_overlap.build_reference_set_records(
        db=db,
        descriptor_distance_matrix=descriptor_distance_matrix,
        tanimoto_distance_matrix=tanimoto_distance_matrix,
        targets=targets,
        max_k_values=max_k_values,
    )
    pairwise_records = reference_overlap.build_pairwise_records(reference_records)
    literature_by_index = load_literature_sets(args.literature_doi)
    literature_stats = build_literature_stats(reference_records, literature_by_index)
    total_rows = build_total_rows(pairwise_records, literature_stats)

    output_csv = args.output_csv or args.output_dir / "combined_overlap_summary.csv"
    copied_code_path = sync_code_to_output_dir(output_csv.parent)
    write_total_csv(output_csv, total_rows)

    print(f"Targets: {len(targets)}")
    print(f"Reference-set rows: {len(reference_records)}")
    omitted_pairwise_groups = count_pairwise_groups(pairwise_records) - len(
        [row for row in total_rows if row["metric_type"] == "pairwise_molecule_overlap"]
    )
    print(
        "Pairwise summary rows: "
        f"{len(pairwise_records)} raw comparisons -> "
        f"{count_pairwise_groups(pairwise_records) - omitted_pairwise_groups} grouped rows "
        f"(omitted fully identical groups: {omitted_pairwise_groups})"
    )
    print(f"Same-literature summary rows: {len(literature_stats)}")
    print(f"Output folder: {output_csv.parent.resolve()}")
    print(f"Total CSV: {output_csv.resolve()}")
    print(f"Copied code: {copied_code_path.resolve()}")


if __name__ == "__main__":
    main()
