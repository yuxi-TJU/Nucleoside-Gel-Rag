from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(gettempdir()) / "qla-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 10.0,
        "axes.linewidth": 0.8,
        "legend.frameon": True,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATA_DIR = SCRIPT_DIR.parent / "3a_similarity_distributions" / "data"
FALLBACK_DIR = SCRIPT_DIR.parent / "3a_similarity_distributions"
if str(FALLBACK_DIR) not in sys.path:
    sys.path.insert(0, str(FALLBACK_DIR))
from plot_combined_strategy_similarity import ensure_dataset_csv  # pyright: ignore[reportMissingImports]  # noqa: E402
DEFAULT_BASE_INPUT = SHARED_DATA_DIR / "text-embedding-3-large_physicochemical_descriptor_similarity_rows.csv"
DEFAULT_MORGAN_INPUT = SHARED_DATA_DIR / "text-embedding-3-large_morgan_fingerprint_similarity_rows.csv"
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
# Encoder names are distinct from the prediction models in MODEL_ORDER.
EMBEDDING_CONFIGS = {
    "text-embedding-3-large": {
        "base_input": DEFAULT_BASE_INPUT,
        "morgan_input": DEFAULT_MORGAN_INPUT,
        "score_column": "S_final",
    },
    "gemini-embedding-2": {
        "base_input": SHARED_DATA_DIR / "gemini-embedding-2_physicochemical_descriptor_similarity_rows.csv",
        "morgan_input": None,
        "score_column": "S_final",
    },
    "semcse": {
        "base_input": SHARED_DATA_DIR / "semcse_physicochemical_descriptor_alignment_zscore_D_final_rows.csv",
        "morgan_input": None,
        "score_column": "alignment_zscore",
    },
}

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

STRATEGIES = [0, 1, 2, 3]
ADJACENT_PAIRS = [(0, 1), (1, 2), (2, 3)]
PAIR_LABELS = {
    (0, 1): "S0\u2192S1",
    (1, 2): "S1\u2192S2",
    (2, 3): "S2\u2192S3",
}

TRANSITION_ORDER = ["C->W", "W->W", "C->C", "W->C"]
TRANSITION_LABELS = {
    "C->W": "C\u2192W",
    "W->W": "W\u2192W",
    "C->C": "C\u2192C",
    "W->C": "W\u2192C",
}
TRANSITION_COLORS = {
    "C->W": "#CD4646",
    "W->W": "#6F7780",
    "C->C": "#05918F",
    "W->C": "#3E6FA3",
}
TRANSITION_X_OFFSETS = {
    "C->W": 0.0,
    "W->W": 0.0,
    "C->C": 0.0,
    "W->C": 0.0,
}
ADJACENT_ENDPOINT_GAP = 0.145
LINE_WIDTH_TO_MARKER_DIAMETER = 0.235
BUBBLE_EDGE_WIDTH = 1.20
BUBBLE_EDGE_DARKEN_FACTOR = 0.72
EXPANDED_STRATEGY_POSITIONS = {
    0: -0.20,
    1: 0.93,
    2: 2.07,
    3: 3.20,
}
TWO_PANEL_STRATEGY_POSITIONS = {
    0: 0.00,
    1: 0.65,
    2: 1.30,
    3: 1.95,
}
TWO_PANEL_XLIM = (-0.22, 2.17)
PAIR_CENTERS = {
    (0, 1): 0.0,
    (1, 2): 0.74,
    (2, 3): 1.48,
}
PAIR_HALF_WIDTH = 0.25
THREE_SLOPE_XLIM = (-0.50, 1.98)

LEFT_PANEL = "S0-S3"
RIGHT_PANEL = "S0 + Morgan fingerprint"
PANEL_TITLES = {
    LEFT_PANEL: "S0-S3",
    RIGHT_PANEL: "S0, S1_1/S2_1/S3_1 (Morgan fingerprint)",
}
PANEL_STRATEGY_LABELS = {
    LEFT_PANEL: {0: "S0", 1: "S1", 2: "S2", 3: "S3"},
    RIGHT_PANEL: {0: "S0", 1: "S1_1", 2: "S2_1", 3: "S3_1"},
}

OUTCOME_ORDER = [True, False]
OUTCOME_LABELS = {True: "Correct", False: "Wrong"}
OUTCOME_COLORS = {True: "#3E6FA3", False: "#CD4646"}

PAIR_ID_COLUMNS = ["molecule_id", "experiment_id", "solvent_additive"]
SCORE_COLUMN_CANDIDATES = ["S_final", "semantic_similarity"]
CORRECTNESS_COLUMN_CANDIDATES = [
    "prediction_correct",
    "desired_majority_correct",
    "is_correct",
    "correct",
]

Y_MIN = 0.46
Y_MAX = 0.72
Y_TICKS = np.arange(0.46, Y_MAX + 1e-12, 0.03)
FIGSIZE_ALL5 = (14.6, 4.0)
FIGSIZE_TWO_PANEL = (19.0, 7.8)
XLIM = (-0.45, 3.45)
X_TICK_POSITIONS = STRATEGIES
AXIS_TITLE_FONT_SIZE = 12
Y_AXIS_TITLE_X = 0.038


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot separated adjacent-strategy transition-group slopes."
    )
    parser.add_argument("--base-input", "--input", dest="base_input", type=Path, default=None)
    parser.add_argument("--morgan-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-column", default=None)
    parser.add_argument(
        "--encoder", choices=list(EMBEDDING_CONFIGS), default=None,
        help="Run one encoder; by default run all four. Custom inputs require this option.",
    )
    parser.add_argument("--model-column", default="source_model")
    parser.add_argument("--pair-id-columns", nargs="+", default=PAIR_ID_COLUMNS)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--keep-old-outputs",
        action="store_true",
        help="Compatibility option; existing outputs are now always preserved.",
    )
    return parser.parse_args()


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
            "wrong": False,
            "incorrect": False,
        }
    )


def safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-_.")
    return safe.lower() or "figure"


def display_model_name(model: str) -> str:
    return DISPLAY_MODEL_NAMES.get(model, model[:1].upper() + model[1:])


def ordered_models(models: list[str]) -> list[str]:
    order_index = {model: index for index, model in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda model: (order_index.get(model, 999), model))


def load_rows(
    input_path: Path,
    model_column: str,
    score_column: str,
    pair_id_columns: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    if model_column not in df.columns and "model_name" in df.columns:
        model_column = "model_name"

    resolved_score_column = score_column
    if resolved_score_column not in df.columns:
        raise ValueError(f"{input_path}: missing requested score column {score_column!r}.")

    correctness_column = first_existing_column(df, CORRECTNESS_COLUMN_CANDIDATES)
    if correctness_column is None:
        raise ValueError(
            "No correctness column found. Expected one of: "
            f"{', '.join(CORRECTNESS_COLUMN_CANDIDATES)}."
        )

    pair_columns = [column for column in pair_id_columns if column in df.columns]
    if not pair_columns:
        raise ValueError(
            "No requested pair-id columns exist in the input CSV. "
            f"Requested: {', '.join(pair_id_columns)}."
        )

    required = ["strategy_index", model_column, resolved_score_column, correctness_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    data = df.copy()
    data["source_model"] = data[model_column].astype(str)
    data["strategy_name"] = data["strategy"].astype(str) if "strategy" in data.columns else ""
    data["strategy_index"] = pd.to_numeric(data["strategy_index"], errors="coerce")
    data["similarity_score"] = pd.to_numeric(data[resolved_score_column], errors="coerce")
    data["prediction_correct"] = parse_bool_series(data[correctness_column])

    keep_columns = [
        "source_model",
        "strategy_name",
        "strategy_index",
        "similarity_score",
        "prediction_correct",
        *pair_columns,
    ]
    before_required_filter = len(data)
    required_mask = data[
        [
            "source_model",
            "strategy_index",
            "similarity_score",
            "prediction_correct",
        ]
    ].notna().all(axis=1)
    data = data.loc[required_mask, keep_columns].copy()
    data["strategy_index"] = data["strategy_index"].astype(int)
    data["prediction_correct"] = data["prediction_correct"].astype(bool)
    data.attrs["excluded_required_value_rows"] = before_required_filter - len(data)
    return data, pair_columns, resolved_score_column


def select_panel_rows(
    base_rows: pd.DataFrame,
    morgan_rows: pd.DataFrame,
    panel: str,
) -> pd.DataFrame:
    """Return one normalized S0-S3 sequence for a comparison panel."""
    if panel == LEFT_PANEL:
        selected = base_rows[base_rows["strategy_index"].isin(STRATEGIES)].copy()
        selected["strategy_index"] = selected["strategy_index"].astype(int)
        return selected

    frames = [base_rows[base_rows["strategy_index"].eq(0)].copy()]
    # Historical input tables named the S2_1/S3_1 output directories 2_2/3_3.
    # Accept either source convention while exporting the canonical labels.
    for plot_index, strategy_aliases in enumerate(
        [("Strategy 1_1",), ("Strategy 2_1", "Strategy 2_2"),
         ("Strategy 3_1", "Strategy 3_3")],
        start=1,
    ):
        strategy_name = next((name for name in strategy_aliases
                              if morgan_rows["strategy_name"].eq(name).any()),
                             strategy_aliases[0])
        rows = morgan_rows[morgan_rows["strategy_name"].eq(strategy_name)].copy()
        if rows.empty:
            raise ValueError(f"No rows found for Morgan-fingerprint strategy {strategy_name}.")
        rows["strategy_index"] = plot_index
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def transition_label(start_correct: bool, end_correct: bool) -> str:
    if start_correct and not end_correct:
        return "C->W"
    if not start_correct and not end_correct:
        return "W->W"
    if start_correct and end_correct:
        return "C->C"
    return "W->C"


def bootstrap_endpoint_cis(
    start_values: np.ndarray,
    end_values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float, float]:
    start_values = np.asarray(start_values, dtype=float)
    end_values = np.asarray(end_values, dtype=float)
    finite = np.isfinite(start_values) & np.isfinite(end_values)
    start_values = start_values[finite]
    end_values = end_values[finite]

    if len(start_values) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if len(start_values) == 1 or n_bootstrap <= 0:
        return (
            float(start_values[0]),
            float(start_values[0]),
            float(end_values[0]),
            float(end_values[0]),
        )

    sample_indices = rng.integers(
        0,
        len(start_values),
        size=(n_bootstrap, len(start_values)),
    )
    boot_start = start_values[sample_indices].mean(axis=1)
    boot_end = end_values[sample_indices].mean(axis=1)
    start_low, start_high = np.percentile(boot_start, [2.5, 97.5])
    end_low, end_high = np.percentile(boot_end, [2.5, 97.5])
    return float(start_low), float(start_high), float(end_low), float(end_high)


def build_transition_data(
    df: pd.DataFrame,
    pair_columns: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_columns = ["source_model", *pair_columns]
    score_pivot = df.pivot_table(
        index=index_columns,
        columns="strategy_index",
        values="similarity_score",
        aggfunc="mean",
    )
    correctness_pivot = df.pivot_table(
        index=index_columns,
        columns="strategy_index",
        values="prediction_correct",
        aggfunc="first",
    )

    summary_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    model_values = df.loc[df["source_model"].notna(), "source_model"].astype(str)
    available_models = ordered_models(sorted(model_values.unique()))

    for model in available_models:
        if model not in score_pivot.index.get_level_values("source_model"):
            continue
        model_scores = score_pivot.xs(model, level="source_model")
        model_correctness = correctness_pivot.xs(model, level="source_model")
        for start_strategy, end_strategy in ADJACENT_PAIRS:
            frame = pd.DataFrame(
                {
                    "start_similarity": model_scores.get(start_strategy),
                    "end_similarity": model_scores.get(end_strategy),
                    "start_correct": model_correctness.get(start_strategy),
                    "end_correct": model_correctness.get(end_strategy),
                }
            )
            complete_mask = frame.notna().all(axis=1)
            frame = frame.loc[complete_mask].copy()
            if frame.empty:
                continue

            frame["transition"] = [
                transition_label(bool(start), bool(end))
                for start, end in zip(frame["start_correct"], frame["end_correct"])
            ]
            frame = frame.reset_index()
            for transition in TRANSITION_ORDER:
                group = frame[frame["transition"] == transition]
                if group.empty:
                    continue

                start_values = group["start_similarity"].to_numpy(dtype=float)
                end_values = group["end_similarity"].to_numpy(dtype=float)
                start_low, start_high, end_low, end_high = bootstrap_endpoint_cis(
                    start_values,
                    end_values,
                    rng,
                    n_bootstrap,
                )
                summary_rows.append(
                    {
                        "source_model": model,
                        "display_model": display_model_name(model),
                        "strategy_pair": PAIR_LABELS[(start_strategy, end_strategy)],
                        "start_strategy": start_strategy,
                        "end_strategy": end_strategy,
                        "transition": transition,
                        "transition_label": TRANSITION_LABELS[transition],
                        "start_outcome": OUTCOME_LABELS[
                            bool(group["start_correct"].iloc[0])
                        ],
                        "end_outcome": OUTCOME_LABELS[
                            bool(group["end_correct"].iloc[0])
                        ],
                        "n": int(len(group)),
                        "start_mean": float(np.mean(start_values)),
                        "end_mean": float(np.mean(end_values)),
                        "start_ci_low": start_low,
                        "start_ci_high": start_high,
                        "end_ci_low": end_low,
                        "end_ci_high": end_high,
                    }
                )

                for row in group.itertuples(index=False):
                    member = {
                        "source_model": model,
                        "strategy_pair": PAIR_LABELS[(start_strategy, end_strategy)],
                        "start_strategy": start_strategy,
                        "end_strategy": end_strategy,
                        "transition": transition,
                        "start_similarity": float(row.start_similarity),
                        "end_similarity": float(row.end_similarity),
                        "start_outcome": OUTCOME_LABELS[bool(row.start_correct)],
                        "end_outcome": OUTCOME_LABELS[bool(row.end_correct)],
                    }
                    for column in pair_columns:
                        member[column] = getattr(row, column)
                    member_rows.append(member)

    return pd.DataFrame(summary_rows), pd.DataFrame(member_rows)


def build_panel_transition_data(
    base_rows: pd.DataFrame,
    morgan_rows: pd.DataFrame,
    pair_columns: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    memberships: list[pd.DataFrame] = []
    for panel in [LEFT_PANEL, RIGHT_PANEL]:
        panel_rows = select_panel_rows(base_rows, morgan_rows, panel)
        summary, members = build_transition_data(
            panel_rows,
            pair_columns,
            rng,
            n_bootstrap,
        )
        labels = PANEL_STRATEGY_LABELS[panel]
        for frame in (summary, members):
            frame.insert(0, "panel", panel)
            frame["start_strategy_label"] = frame["start_strategy"].map(labels)
            frame["end_strategy_label"] = frame["end_strategy"].map(labels)
            frame["strategy_pair"] = (
                frame["start_strategy_label"] + "\u2192" + frame["end_strategy_label"]
            )
        summaries.append(summary)
        memberships.append(members)
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(memberships, ignore_index=True),
    )


def point_area(n: int) -> float:
    return 14.0 + 2.45 * float(n)


def line_width(n: int) -> float:
    return LINE_WIDTH_TO_MARKER_DIAMETER * float(np.sqrt(point_area(n)))


def darken_color(color: str) -> tuple[float, float, float, float]:
    red, green, blue, alpha = to_rgba(color)
    return (
        red * BUBBLE_EDGE_DARKEN_FACTOR,
        green * BUBBLE_EDGE_DARKEN_FACTOR,
        blue * BUBBLE_EDGE_DARKEN_FACTOR,
        alpha,
    )


def endpoint_x(
    strategy: int,
    transition: str,
    is_start: bool,
    strategy_positions: dict[int, float] | None = None,
) -> float:
    adjacent_pair_offset = 0.0
    if strategy in {1, 2}:
        adjacent_pair_offset = ADJACENT_ENDPOINT_GAP if is_start else -ADJACENT_ENDPOINT_GAP
    base_x = (
        strategy_positions.get(strategy, float(strategy))
        if strategy_positions is not None
        else float(strategy)
    )
    return base_x + adjacent_pair_offset + TRANSITION_X_OFFSETS[transition]


def style_axis(
    ax: plt.Axes,
    hide_y_axis: bool,
    strategy_positions: dict[int, float] | None = None,
    strategy_labels: dict[int, str] | None = None,
    xlim: tuple[float, float] = XLIM,
) -> None:
    x_tick_positions = [
        strategy_positions.get(strategy, float(strategy))
        if strategy_positions is not None
        else float(strategy)
        for strategy in STRATEGIES
    ]
    ax.grid(axis="y", color="#E6EAEE", linewidth=0.85, zorder=0)
    # for boundary in x_tick_positions:
    #     ax.axvline(boundary, color="#E8ECEF", linewidth=0.72, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    if hide_y_axis:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=True, labelleft=False)
    ax.tick_params(axis="both", colors="black")
    ax.tick_params(axis="x", labelsize=AXIS_TITLE_FONT_SIZE)
    ax.set_xlim(*xlim)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(Y_TICKS)
    ax.set_xticks(x_tick_positions)
    labels = strategy_labels or PANEL_STRATEGY_LABELS[LEFT_PANEL]
    ax.set_xticklabels([labels[strategy] for strategy in STRATEGIES])


def draw_endpoint(
    ax: plt.Axes,
    x: float,
    y: float,
    low: float,
    high: float,
    correct: bool,
    n: int,
    edge_color: str,
) -> None:
    ax.errorbar(
        [x],
        [y],
        yerr=np.array([[y - low], [high - y]]),
        fmt="none",
        ecolor="#E7E7E7",
        elinewidth=0.9,
        capsize=2.1,
        capthick=0.75,
        alpha=0.95,
        zorder=3,
    )
    ax.scatter(
        [x],
        [y],
        s=point_area(n),
        facecolors=[to_rgba(OUTCOME_COLORS[correct], 0.70)],
        edgecolors=[darken_color(edge_color)],
        linewidth=BUBBLE_EDGE_WIDTH,
        zorder=4,
    )


def draw_model_panel(
    ax: plt.Axes,
    model: str,
    summary: pd.DataFrame,
    strategy_positions: dict[int, float] | None = None,
    panel: str | None = None,
) -> None:
    model_summary = summary[summary["source_model"].eq(model)]
    if panel is not None and "panel" in model_summary.columns:
        model_summary = model_summary[model_summary["panel"].eq(panel)]
    for transition in TRANSITION_ORDER:
        group = model_summary[model_summary["transition"].eq(transition)].sort_values(
            ["start_strategy", "end_strategy"]
        )
        color = TRANSITION_COLORS[transition]
        for row in group.itertuples(index=False):
            start_x = endpoint_x(
                int(row.start_strategy),
                transition,
                is_start=True,
                strategy_positions=strategy_positions,
            )
            end_x = endpoint_x(
                int(row.end_strategy),
                transition,
                is_start=False,
                strategy_positions=strategy_positions,
            )
            ax.plot(
                [start_x, end_x],
                [row.start_mean, row.end_mean],
                color=color,
                linewidth=line_width(int(row.n)),
                alpha=0.86,
                solid_capstyle="round",
                zorder=2,
            )
            draw_endpoint(
                ax,
                start_x,
                float(row.start_mean),
                float(row.start_ci_low),
                float(row.start_ci_high),
                row.start_outcome == "Correct",
                int(row.n),
                color,
            )
            draw_endpoint(
                ax,
                end_x,
                float(row.end_mean),
                float(row.end_ci_low),
                float(row.end_ci_high),
                row.end_outcome == "Correct",
                int(row.n),
                color,
            )


def outcome_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=6.2,
            markerfacecolor=to_rgba(OUTCOME_COLORS[outcome], 0.70),
            markeredgecolor=darken_color(OUTCOME_COLORS[outcome]),
            markeredgewidth=BUBBLE_EDGE_WIDTH,
            label=OUTCOME_LABELS[outcome],
        )
        for outcome in OUTCOME_ORDER
    ]


def transition_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=TRANSITION_COLORS[transition],
            linewidth=2.2,
            solid_capstyle="round",
            label=TRANSITION_LABELS[transition],
        )
        for transition in TRANSITION_ORDER
    ]


def size_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="#6F7780",
            linewidth=line_width(n),
            solid_capstyle="round",
            marker="o",
            markersize=float(np.sqrt(point_area(n))),
            markerfacecolor=to_rgba("#7D8790", 0.70),
            markeredgecolor=darken_color("#6F7780"),
            markeredgewidth=BUBBLE_EDGE_WIDTH,
            label=str(n),
        )
        for n in [10, 30, 60]
    ]


def style_legend(legend: matplotlib.legend.Legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D7DDE3")
    frame.set_linewidth(0.8)
    frame.set_alpha(0.88)


def add_legends(axes: np.ndarray) -> None:
    legend_y = 0.035

    transition_legend = axes[2].legend(
        handles=transition_handles(),
        loc="lower right",
        bbox_to_anchor=(0.985, legend_y),
        frameon=True,
        fontsize=9.0,
        ncol=2,
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    style_legend(transition_legend)
    axes[2].add_artist(transition_legend)

    outcome_legend = axes[3].legend(
        handles=outcome_handles(),
        loc="lower right",
        bbox_to_anchor=(0.985, legend_y),
        frameon=True,
        fontsize=9.0,
        ncol=1,
        handlelength=1.2,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    style_legend(outcome_legend)
    axes[3].add_artist(outcome_legend)

    size_legend = axes[4].legend(
        handles=size_handles(),
        loc="lower right",
        bbox_to_anchor=(0.985, legend_y),
        frameon=True,
        fontsize=9.0,
        ncol=3,
        columnspacing=1.25,
        handleheight=2.0,
        handlelength=1.65,
        handletextpad=0.55,
        labelspacing=0.45,
        borderpad=0.80,
        borderaxespad=0.0,
    )
    style_legend(size_legend)


def validate_summary(summary: pd.DataFrame, expected_panel_count: int = 1) -> None:
    missing_models = [
        model for model in MODEL_ORDER if model not in set(summary["source_model"])
    ]
    if missing_models:
        raise ValueError(
            "The all-five-model layout requires these missing models: "
            f"{', '.join(missing_models)}."
        )
    expected_pairs = len(MODEL_ORDER) * len(ADJACENT_PAIRS) * expected_panel_count
    pair_columns = ["source_model", "start_strategy", "end_strategy"]
    if "panel" in summary.columns:
        pair_columns.insert(0, "panel")
    available_pairs = summary[pair_columns].drop_duplicates()
    if len(available_pairs) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} model/pair combinations, found "
            f"{len(available_pairs)}."
        )


def pair_endpoint_x(start_strategy: int, end_strategy: int, is_start: bool) -> float:
    center = PAIR_CENTERS[(start_strategy, end_strategy)]
    return center - PAIR_HALF_WIDTH if is_start else center + PAIR_HALF_WIDTH


def style_three_slope_axis(
    ax: plt.Axes,
    hide_y_axis: bool,
    panel: str,
) -> None:
    ax.grid(axis="y", color="#E6EAEE", linewidth=0.85, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    if hide_y_axis:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=True, labelleft=False)
    ax.tick_params(axis="both", colors="black")
    ax.tick_params(axis="x", labelsize=AXIS_TITLE_FONT_SIZE)
    ax.set_xlim(*THREE_SLOPE_XLIM)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(Y_TICKS)
    ax.set_xticks(list(PAIR_CENTERS.values()))
    labels = PANEL_STRATEGY_LABELS[panel]
    ax.set_xticklabels(
        [f"{labels[start]}\u2192{labels[end]}" for start, end in ADJACENT_PAIRS]
    )


def draw_three_slope_panel(
    ax: plt.Axes,
    model: str,
    summary: pd.DataFrame,
    panel: str,
) -> None:
    model_summary = summary[
        summary["source_model"].eq(model) & summary["panel"].eq(panel)
    ]
    for transition in TRANSITION_ORDER:
        group = model_summary[model_summary["transition"].eq(transition)].sort_values(
            ["start_strategy", "end_strategy"]
        )
        color = TRANSITION_COLORS[transition]
        for row in group.itertuples(index=False):
            start_x = pair_endpoint_x(
                int(row.start_strategy), int(row.end_strategy), is_start=True
            )
            end_x = pair_endpoint_x(
                int(row.start_strategy), int(row.end_strategy), is_start=False
            )
            ax.plot(
                [start_x, end_x],
                [row.start_mean, row.end_mean],
                color=color,
                linewidth=line_width(int(row.n)),
                alpha=0.86,
                solid_capstyle="round",
                zorder=2,
            )
            draw_endpoint(
                ax,
                start_x,
                float(row.start_mean),
                float(row.start_ci_low),
                float(row.start_ci_high),
                row.start_outcome == "Correct",
                int(row.n),
                color,
            )
            draw_endpoint(
                ax,
                end_x,
                float(row.end_mean),
                float(row.end_ci_low),
                float(row.end_ci_high),
                row.end_outcome == "Correct",
                int(row.n),
                color,
            )


def plot_three_slope_comparisons(
    summary: pd.DataFrame,
    output_dir: Path,
    encoder: str = "text-embedding-3-large",
    panel: str = LEFT_PANEL,
    scale_summary: pd.DataFrame | None = None,
) -> Path:
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=FIGSIZE_ALL5, sharey=True)
    axes = np.ravel(axes)
    for model_index, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        style_three_slope_axis(
            ax,
            hide_y_axis=model_index > 0,
            panel=panel,
        )
        draw_three_slope_panel(
            ax,
            model,
            summary,
            panel=panel,
        )
        if panel == RIGHT_PANEL:
            ax.tick_params(axis="x", labelsize=9)
        ax.set_title(display_model_name(model), fontsize=13, weight="bold", pad=8)

    axes[0].set_ylabel(
        score_axis_label(encoder),
        fontsize=AXIS_TITLE_FONT_SIZE,
        color="black",
    )
    add_legends(axes)
    fig.tight_layout(rect=[0.018, 0.065, 1.0, 0.93], w_pad=1.1)

    analysis_type = (
        "morgan-fingerprint"
        if panel == RIGHT_PANEL
        else "physicochemical-descriptor_alignment-zscore_D-final"
        if encoder == "semcse"
        else "physicochemical-descriptor"
    )
    output_path = output_dir / (
        f"{encoder}_{analysis_type}_transition-group-mean-trajectories_all-models.png"
    )
    # Match the legacy figure's fixed y-axis for both OpenAI strategy panels.
    if encoder != "text-embedding-3-large":
        adjust_score_axes(axes, summary if scale_summary is None else scale_summary, encoder)
    fig.savefig(output_path, dpi=600, facecolor="white")
    plt.close(fig)
    return output_path


def plot_three_slope_comparisons_two_panel(
    summary: pd.DataFrame,
    output_dir: Path,
    encoder: str = "text-embedding-3-large",
) -> Path:
    fig, axes = plt.subplots(
        2,
        len(MODEL_ORDER),
        figsize=FIGSIZE_TWO_PANEL,
        sharey=True,
        squeeze=False,
    )
    for row_index, panel in enumerate([LEFT_PANEL, RIGHT_PANEL]):
        for col_index, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, col_index]
            style_three_slope_axis(
                ax,
                hide_y_axis=col_index > 0,
                panel=panel,
            )
            ax.tick_params(axis="x", labelsize=10)
            draw_three_slope_panel(
                ax,
                model,
                summary,
                panel=panel,
            )
            ax.set_title(display_model_name(model), fontsize=12.5, weight="bold", pad=8)
            if col_index == 0:
                ax.set_ylabel(score_axis_label(encoder), fontsize=AXIS_TITLE_FONT_SIZE)

    transition_legend = fig.legend(
        handles=transition_handles(),
        loc="center left",
        borderaxespad=0,
        ncol=4,
        fontsize=9.0,
        frameon=True,
    )
    style_legend(transition_legend)
    outcome_legend = fig.legend(
        handles=outcome_handles(),
        loc="center left",
        borderaxespad=0,
        ncol=2,
        fontsize=9.0,
        frameon=True,
    )
    style_legend(outcome_legend)
    size_legend = fig.legend(
        handles=size_handles(),
        loc="center left",
        borderaxespad=0,
        ncol=3,
        fontsize=9.0,
        frameon=True,
    )
    style_legend(size_legend)

    fig.text(0.52, 0.965, PANEL_TITLES[LEFT_PANEL], ha="center",
             fontsize=14, weight="bold")
    fig.text(0.52, 0.525, PANEL_TITLES[RIGHT_PANEL], ha="center",
             fontsize=14, weight="bold")
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.16, top=0.89,
                        wspace=0.13, hspace=0.80)

    # Center the legend group using its rendered widths and equal physical gaps.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legends = [transition_legend, outcome_legend, size_legend]
    widths = [legend.get_window_extent(renderer).width / fig.bbox.width
              for legend in legends]
    gap = 18 / 72 / fig.get_figwidth()
    legend_x = (1 - sum(widths) - gap * (len(legends) - 1)) / 2
    for legend, width in zip(legends, widths):
        legend.set_bbox_to_anchor((legend_x, 0.055), transform=fig.transFigure)
        legend_x += width + gap

    output_path = output_dir / (
        "adjacent_strategy_transition_group_mean_trajectories_"
        f"three_slope_comparisons_two_panel_{encoder}.png"
    )
    # Match the reference figure's display window for the requested encoder.
    # Underlying means and confidence limits remain unchanged, including CIs
    # that extend below the displayed lower bound.
    if encoder != "text-embedding-3-large":
        adjust_score_axes(axes, summary, encoder)
    fig.savefig(output_path, dpi=600, facecolor="white")
    plt.close(fig)
    return output_path


def score_axis_label(encoder: str) -> str:
    if encoder == "semcse":
        return "Mean alignment z-score"
    return "Mean semantic similarity"


def adjust_score_axes(axes, summary: pd.DataFrame, encoder: str) -> None:
    """Keep the original scale when possible, otherwise include all means and CIs."""
    values = summary[["start_mean", "end_mean", "start_ci_low", "start_ci_high",
                      "end_ci_low", "end_ci_high"]].to_numpy(dtype=float)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if encoder == "text-embedding-3-large" and Y_MIN <= low and high <= Y_MAX:
        return
    pad = max(0.02, (high - low) * 0.08)
    lower, upper = np.floor((low - pad) * 20) / 20, np.ceil((high + pad) * 20) / 20
    for ax in np.ravel(axes):
        ax.set_yticks(np.linspace(lower, upper, 6))
        ax.set_ylim(lower, upper)


def write_outputs(
    summary: pd.DataFrame,
    members: pd.DataFrame,
    output_dir: Path,
    encoder: str = "text-embedding-3-large",
) -> tuple[Path, Path]:
    is_morgan = "panel" in summary.columns and set(summary["panel"]) == {RIGHT_PANEL}
    analysis_type = (
        "morgan-fingerprint"
        if is_morgan
        else "physicochemical-descriptor_alignment-zscore_D-final"
        if encoder == "semcse"
        else "physicochemical-descriptor"
    )
    stats_path = output_dir / f"{encoder}_{analysis_type}_transition-group-mean-statistics.csv"
    members_path = output_dir / f"{encoder}_{analysis_type}_transition-group-membership.csv"
    sort_columns = ["source_model", "start_strategy", "transition"]
    if "panel" in summary.columns:
        sort_columns.insert(0, "panel")
    summary.sort_values(
        sort_columns,
        kind="stable",
    ).to_csv(stats_path, index=False, encoding="utf-8-sig")
    members.sort_values(
        sort_columns,
        kind="stable",
    ).to_csv(members_path, index=False, encoding="utf-8-sig")
    return stats_path, members_path


def run_encoder(args: argparse.Namespace, encoder: str) -> None:
    config = EMBEDDING_CONFIGS[encoder]
    base_input_path = (args.base_input or config["base_input"]).resolve()
    morgan_input = args.morgan_input or config["morgan_input"]
    if args.base_input is None:
        base_input_path = ensure_dataset_csv(base_input_path)
    elif not base_input_path.is_file():
        raise FileNotFoundError(base_input_path)
    if morgan_input is not None:
        morgan_input = Path(morgan_input).resolve()
        if args.morgan_input is None:
            morgan_input = ensure_dataset_csv(morgan_input)
        elif not morgan_input.is_file():
            raise FileNotFoundError(morgan_input)
    score_column = args.score_column or config["score_column"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    base_rows, pair_columns, resolved_score_column = load_rows(
        base_input_path, args.model_column, score_column, args.pair_id_columns,
    )
    if morgan_input is not None:
        morgan_rows, morgan_pair_columns, _ = load_rows(
            morgan_input.resolve(), args.model_column, score_column, args.pair_id_columns,
        )
        if pair_columns != morgan_pair_columns:
            raise ValueError("Base and Morgan inputs resolved to different pair-id columns.")
        summary, members = build_panel_transition_data(
            base_rows, morgan_rows, pair_columns, rng, args.bootstrap,
        )
        validate_summary(summary, expected_panel_count=2)
    else:
        panel_rows = select_panel_rows(base_rows, base_rows, LEFT_PANEL)
        summary, members = build_transition_data(panel_rows, pair_columns, rng, args.bootstrap)
        for frame in (summary, members):
            frame.insert(0, "panel", LEFT_PANEL)
            frame["start_strategy_label"] = frame["start_strategy"].map(PANEL_STRATEGY_LABELS[LEFT_PANEL])
            frame["end_strategy_label"] = frame["end_strategy"].map(PANEL_STRATEGY_LABELS[LEFT_PANEL])
        validate_summary(summary)
        print(f"[{encoder}] No Morgan input configured; generating S0-S3 outputs only.")

    for frame in (summary, members):
        frame["encoder"] = encoder
        frame["score_column"] = resolved_score_column
    base_summary = summary[summary["panel"].eq(LEFT_PANEL)].copy()
    base_members = members[members["panel"].eq(LEFT_PANEL)].copy()
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = list(write_outputs(base_summary, base_members, DEFAULT_DATA_DIR, encoder))
    paths.append(plot_three_slope_comparisons(base_summary, output_dir, encoder))
    if morgan_input is not None:
        morgan_summary = summary[summary["panel"].eq(RIGHT_PANEL)].copy()
        morgan_members = members[members["panel"].eq(RIGHT_PANEL)].copy()
        paths.extend(write_outputs(morgan_summary, morgan_members, DEFAULT_DATA_DIR, encoder))
        paths.append(plot_three_slope_comparisons(
            morgan_summary, output_dir, encoder, panel=RIGHT_PANEL,
            scale_summary=base_summary))
    print(f"[{encoder}] Base rows: {len(base_rows)} from {base_input_path}")
    print(f"[{encoder}] Score: {resolved_score_column}; complete adjacent pairs: {len(members)}")
    for path in paths:
        print(f"Output written: {path}")


def main() -> None:
    args = parse_args()
    if args.encoder is None and any(
        value is not None for value in (args.base_input, args.morgan_input, args.score_column)
    ):
        raise ValueError("Custom input/score overrides require --encoder to identify the output model.")
    for encoder in ([args.encoder] if args.encoder else EMBEDDING_CONFIGS):
        run_encoder(args, encoder)


if __name__ == "__main__":
    main()
