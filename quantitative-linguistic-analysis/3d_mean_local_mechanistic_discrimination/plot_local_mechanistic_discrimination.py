from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
SIMILARITY_DIR = SCRIPT_DIR.parent / "3a_similarity_distributions"
if str(SIMILARITY_DIR) not in sys.path:
    sys.path.insert(0, str(SIMILARITY_DIR))
from compact_source_data import DEFAULT_SOURCE_TABLE  # pyright: ignore[reportMissingImports]  # noqa: E402
from plot_combined_strategy_similarity import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    CACHE_ROOT,
    LocalCacheReader,
    _records_by_group,
    _selected_rounds,
)


DATA_DIR = SCRIPT_DIR / "data"
FIGURE_DIR = SCRIPT_DIR / "figures"

MODEL_ORDER = [
    "grok-4.3",
    "deepseek-v3.2-think",
    "gemini-3.1-flash-lite",
    "gpt-4o",
    "llama-4-scout",
]
MODEL_LABELS = {
    "grok-4.3": "Grok 4.3",
    "deepseek-v3.2-think": "DeepSeek-V3.2",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gpt-4o": "GPT-4o",
    "llama-4-scout": "Llama 4 Scout",
}

DATASETS = [
    {
        "stem": "text-embedding-3-large_physicochemical-descriptor_candidates10",
        "strategies": ["S0", "S1", "S2", "S3"],
        "ylabel": "Mean local discrimination",
    },
    {
        "stem": "text-embedding-3-large_morgan-fingerprint_candidates10",
        "strategies": ["S0", "S1_1", "S2_1", "S3_1"],
        "ylabel": "Mean local discrimination",
    },
    {
        "stem": "gemini-embedding-2_physicochemical-descriptor_candidates10",
        "strategies": ["S0", "S1", "S2", "S3"],
        "ylabel": "Mean local discrimination",
    },
    {
        "stem": "semcse_physicochemical-descriptor_alignment-zscore_D-final_candidates10",
        "strategies": ["S0", "S1", "S2", "S3"],
        "ylabel": "Mean local z-score discrimination",
    },
]

COLORS = {"Correct": "#3E6FA3", "Wrong": "#CD4646"}
OFFSETS = {"Correct": -0.095, "Wrong": 0.095}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260912


DESCRIPTOR_PATH: Path = SCRIPT_DIR / "input_data" / "descriptor_24d_values.csv"
SAMPLE_COLUMNS = [
    "model", "strategy", "molecule_id", "experiment_id", "round",
    "correctness", "local_score", "observation_level", "selected_round_count",
    "metric", "score", "requested_candidates", "actual_candidates_min",
    "actual_candidates_max",
]


def _sample_references(groups: dict) -> tuple[list[tuple[int, int]], list[str]]:
    lookup = {}
    for records in groups.values():
        first = records[0]
        lookup[(first.molecule_id, first.experiment_id)] = first.reference_text
    keys = sorted(lookup)
    return keys, [lookup[key] for key in keys]


def _local_candidates(
    sample_keys: list[tuple[int, int]], reference_texts: list[str]
) -> list[np.ndarray]:
    if not DESCRIPTOR_PATH.is_file():
        raise FileNotFoundError(f"Local descriptor input is missing: {DESCRIPTOR_PATH}")
    descriptor = pd.read_csv(DESCRIPTOR_PATH).set_index("molecule_index")
    columns = list(descriptor.columns[2:])
    if len(columns) != 24:
        raise ValueError(f"Expected 24 descriptor columns, found {len(columns)}")
    values = descriptor[columns].to_numpy(float)
    scale = values.std(axis=0)
    scale[scale == 0] = 1
    standardized = (values - values.mean(axis=0)) / scale
    lookup = {
        int(molecule): vector
        for molecule, vector in zip(descriptor.index, standardized)
    }
    target_vectors = np.stack([lookup[molecule] for molecule, _ in sample_keys])
    distances = np.linalg.norm(
        target_vectors[:, None] - target_vectors[None, :], axis=2
    )
    candidates: list[np.ndarray] = []
    for index in range(len(sample_keys)):
        eligible = [
            other for other in range(len(sample_keys))
            if other != index and reference_texts[other] != reference_texts[index]
        ]
        cutoff = sorted(distances[index, eligible])[9]
        selected = [
            other for other in eligible
            if distances[index, other] <= cutoff + 1e-10
        ]
        candidates.append(np.asarray(
            sorted(selected, key=lambda other: (distances[index, other], other)),
            dtype=int,
        ))
    return candidates


def _base_fields(
    model: str,
    strategy: str,
    key: tuple[int, int],
    majority_correct: bool,
    candidate_counts: np.ndarray,
) -> dict[str, Any]:
    return {
        "model": model,
        "strategy": strategy,
        "molecule_id": key[0],
        "experiment_id": key[1],
        "correctness": "Correct" if majority_correct else "Wrong",
        "requested_candidates": 10,
        "actual_candidates_min": int(candidate_counts.min()),
        "actual_candidates_max": int(candidate_counts.max()),
    }


def build_sample_scores(
    backend: str,
    strategies: list[str],
    *,
    source_table: Path = DEFAULT_SOURCE_TABLE,
    cache_root: Path = CACHE_ROOT,
) -> pd.DataFrame:
    groups = _records_by_group(Path(source_table), set(strategies))
    sample_keys, references = _sample_references(groups)
    reference_index = {key: index for index, key in enumerate(sample_keys)}
    candidates = _local_candidates(sample_keys, references)
    candidate_counts = np.asarray([len(item) for item in candidates])
    reader = LocalCacheReader(backend, cache_root)
    rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        for key in sample_keys:
            target = reference_index[key]
            local = candidates[target]
            for strategy in strategies:
                records = groups[(model, strategy, key[0], key[1])]
                majority_correct, selected = _selected_rounds(records)
                base = _base_fields(
                    model, strategy, key, majority_correct, candidate_counts
                )
                if backend == "text":
                    for record in records:
                        target_score = reader.score(
                            record.output_text, references[target]
                        )[2]
                        alternative_scores = [
                            reader.score(record.output_text, references[index])[2]
                            for index in local
                        ]
                        rows.append({
                            **base,
                            "round": record.round_number,
                            "local_score": target_score - float(np.mean(alternative_scores)),
                            "observation_level": "round",
                            "selected_round_count": np.nan,
                            "metric": "D_local",
                            "score": "S_final",
                        })
                    continue

                if backend == "gemini":
                    score_vectors = np.asarray([
                        [reader.score(record.output_text, reference)[2]
                         for reference in references]
                        for record in selected
                    ]).mean(axis=0)
                    local_score = float(
                        score_vectors[target] - score_vectors[local].mean()
                    )
                    metric = "D_local"
                    score_name = "gemini-embedding-2"
                elif backend == "semcse":
                    distances = np.asarray([
                        [reader.score(record.output_text, reference)[2]
                         for reference in references]
                        for record in selected
                    ]).mean(axis=0)
                    count = len(distances)
                    null_means = (distances.sum() - distances) / (count - 1)
                    null_variances = (
                        (np.square(distances).sum() - np.square(distances))
                        / (count - 1)
                        - np.square(null_means)
                    )
                    null_stds = np.sqrt(np.maximum(null_variances, 0))
                    if np.any(null_stds <= 0):
                        raise ValueError("SemCSE null standard deviation is zero")
                    zscores = (null_means - distances) / null_stds
                    local_score = float(zscores[target] - zscores[local].mean())
                    metric = "D_local_zscore"
                    score_name = "semcse-null-zscore_D_final"
                else:
                    raise ValueError(f"Unknown backend: {backend}")
                rows.append({
                    **base,
                    "round": np.nan,
                    "local_score": local_score,
                    "observation_level": "target",
                    "selected_round_count": len(selected),
                    "metric": metric,
                    "score": score_name,
                })

    frame = pd.DataFrame(rows)
    expected = len(MODEL_ORDER) * len(strategies) * 130 * (3 if backend == "text" else 1)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} sample-score rows, found {len(frame)}")
    return frame[SAMPLE_COLUMNS]


SAMPLE_FALLBACKS = {
    "text-embedding-3-large_physicochemical-descriptor_candidates10": {
        "backend": "text",
        "strategies": ["S0", "S1", "S2", "S3"],
        "cache": "text-embedding-3-large",
        "precompute": "precompute_text_embedding_3_large_reference_embeddings.py",
    },
    "text-embedding-3-large_morgan-fingerprint_candidates10": {
        "backend": "text",
        "strategies": ["S0", "S1_1", "S2_1", "S3_1"],
        "cache": "text-embedding-3-large",
        "precompute": "precompute_text_embedding_3_large_reference_embeddings.py",
    },
    "gemini-embedding-2_physicochemical-descriptor_candidates10": {
        "backend": "gemini",
        "strategies": ["S0", "S1", "S2", "S3"],
        "cache": "gemini-embedding-2",
        "precompute": "precompute_gemini_embedding_2_reference_embeddings.py",
    },
    "semcse_physicochemical-descriptor_alignment-zscore_D-final_candidates10": {
        "backend": "semcse",
        "strategies": ["S0", "S1", "S2", "S3"],
        "cache": "CLAUSE-Bielefeld_SemCSE",
        "precompute": "precompute_semcse_reference_embeddings.py",
    },
}


def ensure_sample_score_csv(stem: str) -> Path:
    destination = DATA_DIR / f"{stem}_sample-scores.csv"
    if destination.is_file():
        return destination
    if stem not in SAMPLE_FALLBACKS:
        raise ValueError(f"No candidates10 local-cache fallback for: {stem}")
    config = SAMPLE_FALLBACKS[stem]
    cache_dir = CACHE_ROOT / str(config["cache"]) / "documents"
    if not cache_dir.is_dir() or next(cache_dir.glob("*.pt"), None) is None:
        raise FileNotFoundError(
            f"Required sample-score CSV is missing: {destination}\n"
            f"Folder-local cache is missing or empty: {cache_dir}\n"
            "Generate it from 3a_similarity_distributions/source_data with:\n"
            f"    python {config['precompute']}"
        )
    # Avoid requiring PyTorch when the bundled sample-score CSV already exists.
    frame = build_sample_scores(
        str(config["backend"]),
        list(config["strategies"]),
        cache_root=CACHE_ROOT,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination



def load_sample_scores(path: Path, strategies: list[str]) -> pd.DataFrame:
    required = {
        "model", "strategy", "molecule_id", "experiment_id", "correctness",
        "local_score", "observation_level", "selected_round_count", "metric",
        "score", "requested_candidates", "actual_candidates_min",
        "actual_candidates_max",
    }
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if set(pd.to_numeric(frame["requested_candidates"], errors="raise")) != {10}:
        raise ValueError(f"{path.name} must contain candidates10 results only")
    if set(frame["model"].astype(str)) != set(MODEL_ORDER):
        raise ValueError(f"{path.name} does not contain the expected five models")
    if set(frame["strategy"].astype(str)) != set(strategies):
        raise ValueError(f"Unexpected strategies in {path.name}")
    if frame["local_score"].isna().any():
        raise ValueError(f"{path.name} contains missing local scores")
    levels = set(frame["observation_level"].astype(str))
    if levels not in ({"round"}, {"target"}):
        raise ValueError(f"Unexpected observation level in {path.name}: {levels}")
    return frame


def target_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert round- or target-level exports to one score per experimental target."""
    keys = ["model", "strategy", "molecule_id", "experiment_id"]
    level = str(frame["observation_level"].iloc[0])
    if level == "round":
        round_summary = frame.assign(
            is_correct=frame["correctness"].eq("Correct")
        ).groupby(keys, as_index=False).agg(
            n_all_rounds=("round", "size"),
            n_correct_rounds=("is_correct", "sum"),
        )
        if not round_summary["n_all_rounds"].eq(3).all():
            raise ValueError("Round-level data must contain exactly three rounds per target")
        round_summary["majority_correctness"] = np.where(
            round_summary["n_correct_rounds"].ge(2), "Correct", "Wrong"
        )
        selected = frame.merge(
            round_summary[keys + ["majority_correctness"]],
            on=keys,
            validate="many_to_one",
        )
        selected = selected[
            selected["correctness"].eq(selected["majority_correctness"])
        ]
        targets = selected.groupby(
            keys + ["majority_correctness"], as_index=False
        ).agg(local_score=("local_score", "mean"), n_rounds=("round", "size"))
        return targets.rename(columns={"majority_correctness": "correctness"})

    targets = frame[
        keys + ["correctness", "local_score", "selected_round_count"]
    ].copy()
    targets["n_rounds"] = pd.to_numeric(
        targets.pop("selected_round_count"), errors="raise"
    ).astype(int)
    if targets.duplicated(keys).any():
        raise ValueError("Target-level data contains duplicate targets")
    return targets


def summarize_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """Recalculate group means and molecule-cluster bootstrap confidence intervals."""
    targets = target_scores(frame)
    cohort_sizes = targets.groupby(["model", "strategy"]).size()
    if not cohort_sizes.eq(130).all():
        raise ValueError("Every model/strategy must contain 130 targets")

    metric = str(frame["metric"].iloc[0])
    score = str(frame["score"].iloc[0])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for (model, strategy, correctness), group in targets.groupby(
        ["model", "strategy", "correctness"], sort=True
    ):
        clusters = group.groupby("molecule_id")["local_score"].agg(["sum", "count"])
        indices = rng.integers(
            0, len(clusters), size=(BOOTSTRAP_REPLICATES, len(clusters))
        )
        replicates = (
            clusters["sum"].to_numpy()[indices].sum(axis=1)
            / clusters["count"].to_numpy()[indices].sum(axis=1)
        )
        low, high = np.quantile(replicates, [0.025, 0.975])
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "correctness": correctness,
                "metric": metric,
                "score": score,
                "mean": float(group["local_score"].mean()),
                "low": float(low),
                "high": float(high),
                "n_targets": int(len(group)),
                "n_molecules": int(len(clusters)),
                "n_rounds": int(group["n_rounds"].sum()),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "estimator": "mean_of_majority_matching_rounds_within_target",
                "ci_method": "molecule_cluster_percentile_95",
                "requested_candidates": int(frame["requested_candidates"].iloc[0]),
                "actual_candidates_min": int(frame["actual_candidates_min"].iloc[0]),
                "actual_candidates_max": int(frame["actual_candidates_max"].iloc[0]),
            }
        )
    summary = pd.DataFrame(rows)
    if len(summary) != 40:
        raise ValueError(f"Expected 40 summary rows, found {len(summary)}")
    return summary


def verify_saved_summary(calculated: pd.DataFrame, path: Path) -> None:
    """Check the recalculation against the previously exported 40-row summary."""
    if not path.exists():
        return
    stored = pd.read_csv(path)
    keys = ["model", "strategy", "correctness"]
    check = calculated.merge(
        stored,
        on=keys,
        suffixes=("_calculated", "_stored"),
        validate="one_to_one",
    )
    for column in ["mean", "low", "high"]:
        if not np.allclose(
            check[f"{column}_calculated"],
            check[f"{column}_stored"],
            rtol=1e-10,
            atol=1e-12,
        ):
            difference = np.max(
                np.abs(check[f"{column}_calculated"] - check[f"{column}_stored"])
            )
            raise ValueError(f"{path.name}: {column} differs by {difference}")


def choose_axis_limits(frame: pd.DataFrame) -> tuple[float, float, float]:
    low = min(0.0, float(frame["low"].min()))
    high = max(0.0, float(frame["high"].max()))
    span = max(high - low, 0.02)
    padding = 0.14 * span
    lower = np.floor((low - padding) / 0.02) * 0.02
    upper = np.ceil((high + padding) / 0.02) * 0.02
    raw_step = (upper - lower) / 7
    tick_step = next(
        step for step in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
        if step >= raw_step
    )
    return float(lower), float(upper), tick_step


def draw_dataset(config: dict[str, object], *, plot_only: bool = False) -> Path:
    stem = str(config["stem"])
    strategies = list(config["strategies"])
    summary_path = DATA_DIR / f"{stem}.csv"
    if plot_only:
        frame = pd.read_csv(summary_path)
    else:
        sample_path = ensure_sample_score_csv(stem)
        samples = load_sample_scores(sample_path, strategies)
        frame = summarize_samples(samples)
        verify_saved_summary(frame, summary_path)
        frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Summary recalculated: {summary_path}")
    lower, upper, tick_step = choose_axis_limits(frame)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(14.6, 4.55), sharey=True)
    # Preserve the original panel/axis height after removing the figure title.
    fig.subplots_adjust(left=0.068, right=0.99, bottom=0.16, top=0.77, wspace=0.065)

    for panel_index, (model, ax) in enumerate(zip(MODEL_ORDER, axes)):
        for correctness in ["Correct", "Wrong"]:
            values = (
                frame[(frame["model"] == model) & (frame["correctness"] == correctness)]
                .set_index("strategy")
                .loc[strategies]
            )
            x = np.arange(len(strategies), dtype=float) + OFFSETS[correctness]
            color = COLORS[correctness]
            ax.vlines(x, values["low"], values["high"], color=color, alpha=0.92, linewidth=1)
            ax.hlines(values["low"], x - 0.05, x + 0.05, color=color, linewidth=1)
            ax.hlines(values["high"], x - 0.05, x + 0.05, color=color, linewidth=1)
            ax.plot(x, values["mean"], color=color, alpha=0.78, linewidth=1.5)
            ax.scatter(
                x,
                values["mean"],
                s=14 + 2.45 * values["n_targets"],
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            for x_value, row in zip(x, values.itertuples()):
                direction = 1 if correctness == "Correct" else -1
                ax.annotate(
                    str(int(row.n_targets)),
                    (x_value, row.mean),
                    xytext=(direction * 6, 6),
                    textcoords="offset points",
                    color=color,
                    fontsize=7.4,
                    ha="left" if direction > 0 else "right",
                    va="bottom",
                )

        ax.set_title(MODEL_LABELS[model], fontweight="bold", pad=8)
        ax.set_xticks(range(len(strategies)), strategies)
        ax.set_xlim(-0.45, len(strategies) - 0.55)
        ax.set_ylim(lower, upper)
        ax.set_yticks(np.arange(lower, upper + tick_step / 2, tick_step))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y", color="#E6EAEE", linewidth=0.85, zorder=0)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1.2, zorder=3)
        if panel_index == 0:
            ax.set_ylabel(str(config["ylabel"]))
            ax.yaxis.set_label_coords(-0.16, 0.5)
        else:
            ax.spines["left"].set_visible(False)

    outcome_handles = [
        plt.Line2D([], [], color=COLORS[label], marker="o", markersize=5.8, linewidth=2, label=label)
        for label in ["Correct", "Wrong"]
    ]
    axes[3].legend(handles=outcome_handles, loc="lower right", frameon=True, fontsize=9)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"{stem}_all-models.png"
    fig.savefig(output_path, dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(output_path) as rendered:
        if rendered.format != "PNG" or not all(abs(value - 600) < 0.1 for value in rendered.info["dpi"]):
            raise ValueError(f"Unexpected output format or DPI: {output_path}")
    print(f"Figure written: {output_path}")
    return output_path


def main() -> None:
    for config in DATASETS:
        draw_dataset(config)


if __name__ == "__main__":
    main()
