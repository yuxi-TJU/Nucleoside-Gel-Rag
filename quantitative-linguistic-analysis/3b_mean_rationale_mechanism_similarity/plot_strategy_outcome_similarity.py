from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Arial", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "lines.linewidth": 2.0,
    }
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATA_DIR = SCRIPT_DIR.parent / "3a_similarity_distributions" / "data"
FALLBACK_DIR = SCRIPT_DIR.parent / "3a_similarity_distributions"
if str(FALLBACK_DIR) not in sys.path:
    sys.path.insert(0, str(FALLBACK_DIR))
from plot_combined_strategy_similarity import ensure_dataset_csv  # pyright: ignore[reportMissingImports]  # noqa: E402
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
DEFAULT_SCORE_COLUMN = "S_final"
DEFAULT_PAIR_ID_COLUMNS = ["molecule_id", "experiment_id", "solvent_additive"]


DEFAULT_EMBEDDINGS = [
    {
        "encoder": "text-embedding-3-large",
        "rows": SHARED_DATA_DIR / "text-embedding-3-large_physicochemical_descriptor_similarity_rows.csv",
        "score_column": "S_final",
        "score_multiplier": 1.0,
        "output_suffix": "physicochemical-descriptor",
        "strategy_labels": ["S0", "S1", "S2", "S3"],
    },
    {
        "encoder": "text-embedding-3-large",
        "rows": SHARED_DATA_DIR / "text-embedding-3-large_morgan_fingerprint_similarity_rows.csv",
        "score_column": "S_final",
        "score_multiplier": 1.0,
        "output_suffix": "morgan-fingerprint",
        "strategy_labels": ["S0", "S1_1", "S2_1", "S3_1"],
    },
    {
        "encoder": "gemini-embedding-2",
        "rows": SHARED_DATA_DIR / "gemini-embedding-2_physicochemical_descriptor_similarity_rows.csv",
        "score_column": "S_final",
        "score_multiplier": 1.0,
        "output_suffix": "physicochemical-descriptor",
        "strategy_labels": ["S0", "S1", "S2", "S3"],
    },
    {
        "encoder": "semcse",
        "rows": SHARED_DATA_DIR / "semcse_physicochemical_descriptor_alignment_zscore_D_final_rows.csv",
        "score_column": "alignment_zscore",
        "score_multiplier": 1.0,
        "output_suffix": "physicochemical-descriptor_alignment-zscore_D-final",
        "strategy_labels": ["S0", "S1", "S2", "S3"],
    },
]

MODEL_ORDER = [
    "grok-4.3",
    "deepseek-v3.2-think",
    "gemini-3.1-flash-lite",
    "gpt-4o",
    "llama-4-scout",
]

DISPLAY_MODEL_NAMES = {
    "grok-4.3": "Grok 4.3",
    "deepseek-v3.2-think": "DeepSeek-V3.2",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gpt-4o": "GPT-4o",
    "llama-4-scout": "Llama 4 Scout",
}

OUTCOME_ORDER = [True, False]
OUTCOME_LABELS = {True: "Correct", False: "Wrong"}
OUTCOME_COLORS = {True: "#3E6FA3", False: "#CD4646",}
OUTCOME_X_OFFSETS = {True: -0.095, False: 0.095}
OUTCOME_LABEL_X_OFFSETS = {True: 0.020, False: -0.020}
OUTCOME_LABEL_ALIGNMENTS = {True: "left", False: "right"}
AXIS_TITLE_FONT_SIZE = 12
#Y_AXIS_TITLE_X = 0.038
X_AXIS_TITLE_Y = 0.052


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean semantic similarity by prediction outcome across evidence "
            "strategies."
        )
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--embedding-output-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--model-column", default="source_model")
    parser.add_argument("--pair-id-columns", nargs="+", default=DEFAULT_PAIR_ID_COLUMNS)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--score-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix appended to generated CSV/PNG names before group suffixes.",
    )
    return parser.parse_args()


def safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-_.")
    return safe.lower() or "embedding"


def output_stem(base_stem: str, output_suffix: str | None) -> str:
    suffix = str(output_suffix or "").strip()
    if not suffix:
        return base_stem
    return f"{base_stem}_{suffix}"


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input is not None:
        return args.input.resolve()
    if args.embedding_output_dir is None:
        raise ValueError("Single-embedding mode requires --input or --embedding-output-dir.")
    raise ValueError("Use --input for a custom CSV.")


def infer_encoder_name(path: Path) -> str:
    name = path.name
    if name == "semantic_similarity_gpt_outputs":
        return "text-embedding-3-large"
    if name == "semantic_similarity_gemini_outputs":
        return "gemini-embedding-2"
    if name.startswith("semantic_similarity_") and name.endswith("_outputs"):
        return name.removeprefix("semantic_similarity_").removesuffix("_outputs").replace("_", "-")
    return safe_filename_part(name)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(
        {
            "true": True,
            "t": True,
            "1": True,
            "yes": True,
            "y": True,
            "correct": True,
            "false": False,
            "f": False,
            "0": False,
            "no": False,
            "n": False,
            "incorrect": False,
            "wrong": False,
        }
    )


def load_rows(
    input_path: Path,
    model_column: str,
    score_column: str,
    score_multiplier: float,
    pair_id_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    score_column = first_existing_column(df, [score_column, "S_final", "semantic_similarity"])
    if score_column is None:
        raise ValueError("No similarity score column found.")

    correctness_column = first_existing_column(
        df,
        ["prediction_correct", "desired_majority_correct", "is_correct", "correct"],
    )
    if correctness_column is None:
        raise ValueError("No correctness column found.")

    if model_column not in df.columns and "model_name" in df.columns:
        model_column = "model_name"

    pair_columns = [column for column in pair_id_columns if column in df.columns]
    required = ["strategy_index", model_column, score_column, correctness_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["source_model"] = df[model_column].astype(str)
    df["strategy_index"] = pd.to_numeric(df["strategy_index"], errors="coerce")
    df["raw_score_value"] = pd.to_numeric(df[score_column], errors="coerce")
    df["similarity_score"] = df["raw_score_value"] * score_multiplier
    # Keep S_final explicitly in the detail export, even if a caller chose a
    # different plotting metric (for example, D_final or an alignment score).
    df["S_final"] = (
        pd.to_numeric(df["S_final"], errors="coerce")
        if "S_final" in df.columns
        else np.nan
    )
    df["correct_bool"] = parse_bool_series(df[correctness_column])

    keep_columns = [
        "source_model",
        "strategy_index",
        "similarity_score",
        "raw_score_value",
        "S_final",
        "correct_bool",
        *pair_columns,
    ]
    df = df[keep_columns].dropna(
        subset=["source_model", "strategy_index", "similarity_score", "correct_bool"]
    )
    df["strategy_index"] = df["strategy_index"].astype(int)
    df["correct_bool"] = df["correct_bool"].astype(bool)
    return df, pair_columns


def build_detail_rows(
    df: pd.DataFrame,
    pair_columns: list[str],
    encoder: str,
    score_column: str,
) -> pd.DataFrame:
    """Create one auditable row per molecule-condition-strategy prediction."""
    detail = df.copy()
    detail["strategy"] = "S" + detail["strategy_index"].astype(str)
    detail["prediction_correct"] = detail["correct_bool"].astype(bool)
    detail["prediction_outcome"] = detail["prediction_correct"].map(OUTCOME_LABELS)
    detail["plot_score_column"] = score_column

    ordered_columns = [
        "embedding_model",
        "source_model",
        *pair_columns,
        "strategy_index",
        "strategy",
        "prediction_correct",
        "prediction_outcome",
        "S_final",
        "plot_score_column",
        "raw_score_value",
        "similarity_score",
    ]
    detail.insert(0, "embedding_model", encoder)
    return detail[ordered_columns].sort_values(
        ["source_model", *pair_columns, "strategy_index"],
        kind="stable",
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1 or n_bootstrap <= 0:
        mean = float(np.mean(values))
        return mean, mean

    sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boot_means = values[sample_indices].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def rounded(value: object, digits: int = 6) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(numeric):
        return float("nan")
    return round(numeric, digits)


def build_outcome_summary(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["source_model", "strategy_index", "correct_bool"]
    for group_keys, group in df.groupby(group_columns, sort=True):
        source_model, strategy_index, correct_bool = group_keys
        values = group["similarity_score"].to_numpy(dtype=float)
        mean = float(np.mean(values))
        low, high = bootstrap_mean_ci(values, rng, n_bootstrap)
        rows.append(
            {
                "source_model": source_model,
                "display_model": display_model_name(str(source_model)),
                "strategy_index": int(strategy_index),
                "prediction_correct": bool(correct_bool),
                "prediction_outcome": OUTCOME_LABELS[bool(correct_bool)],
                "n": int(len(values)),
                "mean_similarity": rounded(mean),
                "ci_low": rounded(low),
                "ci_high": rounded(high),
            }
        )
    return pd.DataFrame(rows)


def display_model_name(model_name: str) -> str:
    return DISPLAY_MODEL_NAMES.get(model_name, str(model_name).replace(" ", "-"))


def ordered_models(models: list[str]) -> list[str]:
    order_index = {model: index for index, model in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda model: (order_index.get(model, 999), model))


def marker_area(n: int) -> float:
    return 14.0 + 2.45 * float(n)


def line_width(n: int) -> float:
    return 0.70 + 0.010 * float(n)


def outcome_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            color=OUTCOME_COLORS[outcome],
            marker="o",
            linewidth=2.0,
            markersize=5.8,
            label=OUTCOME_LABELS[outcome],
        )
        for outcome in OUTCOME_ORDER
    ]


def count_handles(max_n: int) -> list[plt.Line2D]:
    candidates = [30, 50, 70, 90]
    return [
        plt.Line2D(
            [0],
            [0],
            color="#7D8790",
            marker="o",
            linewidth=line_width(count),
            markersize=(marker_area(count) ** 0.5) * 0.50,
            label=str(count),
        )
        for count in candidates
    ]


def style_legend(legend: plt.Legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D7DDE3")
    frame.set_linewidth(0.8)
    frame.set_alpha(0.88)


def score_axis_label(encoder: str) -> str:
    if encoder == "semcse":
        return "Mean mechanistic alignment z-score"
    if encoder == "semcse-euclidean":
        return "Mean semantic similarity (-Euclidean distance)"
    if encoder == "semcse-null-percentile":
        return "Mean mechanistic alignment percentile"
    if encoder == "semcse-null-zscore":
        return "Mean mechanistic alignment z-score"
    return "Mean semantic similarity"


def draw_model_panel(
    ax: plt.Axes,
    model: str,
    summary: pd.DataFrame,
    strategies: list[int],
    x_positions: np.ndarray,
    y_min: float,
    y_max: float,
    strategy_labels: list[str],
) -> None:
    model_summary = summary[summary["source_model"] == model]
    ax.grid(axis="y", color="#E6EAEE", linewidth=0.85, zorder=0)
    ax.set_title(display_model_name(model), fontsize=13, weight="bold", pad=8)

    for outcome in OUTCOME_ORDER:
        group = model_summary[model_summary["prediction_correct"] == outcome]
        xs: list[float] = []
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        counts: list[int] = []
        for x_value, strategy_index in zip(x_positions, strategies):
            row = group[group["strategy_index"] == strategy_index]
            if row.empty:
                continue
            row_data = row.iloc[0]
            xs.append(float(x_value) + OUTCOME_X_OFFSETS[outcome])
            means.append(float(row_data["mean_similarity"]))
            lows.append(float(row_data["ci_low"]))
            highs.append(float(row_data["ci_high"]))
            counts.append(int(row_data["n"]))

        if not means:
            continue

        color = OUTCOME_COLORS[outcome]
        linewidth = line_width(max(counts))
        ax.plot(xs, means, color=color, linewidth=linewidth, alpha=0.78, zorder=2)
        for x_value, mean, low, high, n in zip(xs, means, lows, highs, counts):
            ax.errorbar(
                [x_value],
                [mean],
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="none",
                ecolor=color,
                elinewidth=max(1.0, line_width(n) * 0.68),
                capsize=3.0,
                capthick=1.0,
                alpha=0.92,
                zorder=3,
            )
            ax.scatter(
                [x_value],
                [mean],
                s=marker_area(n),
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            ax.text(
                x_value + OUTCOME_LABEL_X_OFFSETS[outcome],
                mean + (0.040 * (y_max - y_min)),
                str(n),
                color=color,
                fontsize=7.4,
                ha=OUTCOME_LABEL_ALIGNMENTS[outcome],
                va="bottom",
                clip_on=False,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategy_labels)
    ax.set_xlim(-0.45, len(strategies) - 0.55)
    ax.set_ylim(y_min, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", colors="black")
    ax.tick_params(axis="x", labelsize=AXIS_TITLE_FONT_SIZE)


def plot_model_group(
    summary: pd.DataFrame,
    output_dir: Path,
    encoder: str,
    models: list[str],
    output_name: str,
    figsize: tuple[float, float],
    w_pad: float = 1.7,
    count_legend_ncol: int = 2,
    y_min_override: float | None = None,
    y_max_override: float | None = None,
    y_tick_step_override: float | None = None,
    legend_panel_indices: tuple[int, int] | None = None,
    y_tick_start_override: float | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = [0, 1, 2, 3]
    label_map = (
        summary[["strategy_index", "strategy_label"]]
        .drop_duplicates()
        .set_index("strategy_index")["strategy_label"]
        .to_dict()
    )
    strategy_labels = [str(label_map.get(strategy, f"S{strategy}")) for strategy in strategies]
    x_positions = np.arange(len(strategies), dtype=float)

    y_values = summary[["mean_similarity", "ci_low", "ci_high"]].to_numpy(dtype=float)
    finite_values = y_values[np.isfinite(y_values)]
    if len(finite_values):
        y_min = float(np.min(finite_values)) - 0.08 * float(np.ptp(finite_values))
        y_max = float(np.max(finite_values)) + 0.10 * float(np.ptp(finite_values))
    else:
        y_min, y_max = 0.0, 1.0
    if y_min_override is not None:
        y_min = y_min_override
    if y_max_override is not None:
        y_max = y_max_override

    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True)
    if len(models) == 1:
        axes = np.array([axes])
    axes_flat = np.ravel(axes)

    for model_index, (ax, model) in enumerate(zip(axes_flat, models)):
        draw_model_panel(
            ax, model, summary, strategies, x_positions, y_min, y_max, strategy_labels
        )
        if y_min_override is not None:
            tick_step = (
                y_tick_step_override
                if y_tick_step_override is not None
                else 0.005 if y_max - y_min <= 0.05 else 0.02
            )
            tick_start = y_min if y_tick_start_override is None else y_tick_start_override
            ax.set_yticks(np.arange(tick_start, y_max + 1e-12, tick_step))
            ax.set_ylim(y_min, y_max)
        if model_index > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=True, labelleft=False)

    max_n = int(summary["n"].max()) if "n" in summary and not summary.empty else 90
    if legend_panel_indices is None:
        outcome_legend_ax = axes_flat[-2] if len(axes_flat) > 1 else axes_flat[-1]
        size_legend_ax = axes_flat[-1]
    else:
        outcome_index, size_index = legend_panel_indices
        outcome_legend_ax = axes_flat[outcome_index]
        size_legend_ax = axes_flat[size_index]

    outcome_legend = outcome_legend_ax.legend(
        handles=outcome_handles(),
        loc="lower right",
        bbox_to_anchor=(0.985, 0.005),
        frameon=True,
        fontsize=9.0,
        ncol=1,
        handlelength=1.7,
    )
    style_legend(outcome_legend)
    outcome_legend_ax.add_artist(outcome_legend)

    size_legend = size_legend_ax.legend(
        handles=count_handles(max_n),
        loc="lower right",
        bbox_to_anchor=(0.985, 0.005),
        frameon=True,
        fontsize=9.0,
        ncol=count_legend_ncol,
        columnspacing=0.8,
        handlelength=1.5,
    )
    style_legend(size_legend)

    axes_flat[0].set_ylabel(
    score_axis_label(encoder),
    fontsize=AXIS_TITLE_FONT_SIZE,
    color="black",
)
    fig.text(
        0.5,
        X_AXIS_TITLE_Y,
        "Evidence strategy",
        ha="center",
        va="center",
        fontsize=AXIS_TITLE_FONT_SIZE,
        color="black",
    )
    # Preserve the original panel/axis height after removing the figure title.
    fig.tight_layout(rect=[0.018, 0.090, 1.0, 0.93], w_pad=w_pad)

    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_summary(
    summary: pd.DataFrame,
    output_dir: Path,
    encoder: str,
    output_suffix: str | None = None,
) -> list[Path]:
    available_models = ordered_models(sorted(summary["source_model"].dropna().astype(str).unique()))
    group_all5 = [model for model in MODEL_ORDER if model in available_models]
    encoder_slug = safe_filename_part(encoder)
    stem = output_stem(f"outcome_similarity_{encoder_slug}", output_suffix)

    paths: list[Path] = []
    if len(group_all5) == len(MODEL_ORDER):
        if encoder == "semcse-cosine":
            y_min_override = 0.900
        elif encoder == "biomedbert":
            y_min_override = 0.61
        elif encoder == "text-embedding-3-large":
            y_min_override = 0.51
        elif encoder == "gemini-embedding-2":
            y_min_override = 0.60
        elif encoder == "semcse-null-percentile":
            y_min_override = 0.55
        else:
            y_min_override = None
        if encoder == "semcse-null-percentile":
            y_max_override = 0.90
        else:
            y_max_override = 0.943 if encoder == "semcse-cosine" else None
        y_tick_step_override = 0.05 if encoder == "semcse-null-percentile" else None
        y_tick_start_override = None
        if encoder == "gemini-embedding-2" and output_suffix == "physicochemical-descriptor":
            y_min_override = 0.75
            y_max_override = 0.85
            y_tick_start_override = 0.76
            y_tick_step_override = 0.02
        legend_panel_indices = (
            (0, 1)
            if encoder in {"biomedbert", "semcse-cosine", "semcse-null-percentile"}
            else None
        )
        paths.append(
            plot_model_group(
                summary,
                output_dir,
                encoder,
                group_all5,
                f"{stem}_all-models.png",
                (14.6, 4.55),
                w_pad=1.1,
                y_min_override=y_min_override,
                y_max_override=y_max_override,
                y_tick_step_override=y_tick_step_override,
                legend_panel_indices=legend_panel_indices,
                y_tick_start_override=y_tick_start_override,
            )
        )
    return paths


def run_embedding(
    input_path: Path,
    output_dir: Path,
    encoder: str,
    score_column: str,
    score_multiplier: float,
    args: argparse.Namespace,
    output_suffix: str | None = None,
    strategy_labels: list[str] | None = None,
) -> list[Path]:
    encoder_slug = safe_filename_part(encoder)
    stem = output_stem(f"outcome_similarity_{encoder_slug}", output_suffix)
    rng = np.random.default_rng(args.seed)
    df, pair_columns = load_rows(
        input_path,
        args.model_column,
        score_column,
        score_multiplier,
        args.pair_id_columns,
    )
    summary = build_outcome_summary(df, rng, args.bootstrap)
    summary.insert(0, "embedding_model", encoder)
    labels = strategy_labels or ["S0", "S1", "S2", "S3"]
    summary["strategy_label"] = summary["strategy_index"].map(dict(enumerate(labels)))

    output_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DEFAULT_DATA_DIR / f"{stem}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_paths = plot_summary(summary, output_dir, encoder, output_suffix)

    print(f"Input rows: {input_path}")
    print(f"Summary written: {summary_path}")
    for plot_path in plot_paths:
        print(f"Plot written: {plot_path}")
    return [summary_path, *plot_paths]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.input is not None or args.embedding_output_dir is not None:
        encoder = args.encoder or infer_encoder_name(
            args.embedding_output_dir if args.embedding_output_dir else args.input
        )
        run_embedding(
            resolve_input_path(args),
            output_dir,
            encoder,
            args.score_column,
            args.score_multiplier,
            args,
            output_suffix=args.output_suffix,
            strategy_labels=["S0", "S1", "S2", "S3"],
        )
        return

    for config in DEFAULT_EMBEDDINGS:
        input_path = Path(config["rows"]).resolve()
        encoder = str(config["encoder"])
        output_suffix = str(config.get("output_suffix", ""))
        input_path = ensure_dataset_csv(input_path)
        run_embedding(
            input_path,
            output_dir,
            encoder,
            str(config["score_column"]),
            float(config["score_multiplier"]),
            args,
            output_suffix=output_suffix,
            strategy_labels=list(config["strategy_labels"]),
        )


if __name__ == "__main__":
    main()
