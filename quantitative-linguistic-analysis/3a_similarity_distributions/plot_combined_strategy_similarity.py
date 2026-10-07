from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except ImportError:
    torch = None

from compact_source_data import (
    DEFAULT_SOURCE_TABLE,
    CompactSourceRecord,
    iter_source_records,
    normalize_result,
    predicted_result,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_ROOT = SCRIPT_DIR / "embedding_cache"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
DATA_DIR = SCRIPT_DIR / "data"
TEXT_EMBEDDING_PREFIX = "text-embedding-3-large"
GEMINI_EMBEDDING_PREFIX = "gemini-embedding-2"
TEXT_DESCRIPTOR_DETAILS = DATA_DIR / f"{TEXT_EMBEDDING_PREFIX}_physicochemical_descriptor_similarity_rows.csv"
TEXT_MORGAN_DETAILS = DATA_DIR / f"{TEXT_EMBEDDING_PREFIX}_morgan_fingerprint_similarity_rows.csv"
GEMINI_DESCRIPTOR_DETAILS = DATA_DIR / f"{GEMINI_EMBEDDING_PREFIX}_physicochemical_descriptor_similarity_rows.csv"


STRATEGIES = range(4)
LABEL_ORDER = [True, False]
CORRECT_COLOR = "#06438A"
INCORRECT_COLOR = "#861B02"
VIOLIN_CORRECT_COLOR = "#06438A"
VIOLIN_INCORRECT_COLOR = "#BE4E0C"
VIOLIN_LABEL_STYLE = {
    True: {"label": "Correct prediction", "color": VIOLIN_CORRECT_COLOR, "y_offset": -0.12},
    False: {"label": "Wrong prediction", "color": VIOLIN_INCORRECT_COLOR, "y_offset": 0.12},
}
INCORRECT_TO_CORRECT_LINE_COLOR = "#042968"
CORRECT_TO_INCORRECT_LINE_COLOR = "#780B07"
SAME_CORRECTNESS_LINE_COLOR = "#CEC9C9"  #"#CEC9C9"
INCORRECT_TO_CORRECT_LINE_ALPHA = 175
CORRECT_TO_INCORRECT_LINE_ALPHA = 175
SAME_CORRECTNESS_LINE_ALPHA = 105
SCORE_COLUMNS = ["semantic_similarity", "S_global", "S_coverage", "S_final"]
CONNECTED_VIOLIN_FILL_ALPHA = 8

MODEL_ORDER = [
    "grok-4.3",
    "deepseek-v3.2-think",
    "gemini-3.1-flash-lite",
    "gpt-4o",
    "llama-4-scout",
]
DESCRIPTOR_STRATEGY_LABELS = {index: f"Strategy {index}" for index in STRATEGIES}
MORGAN_STRATEGY_LABELS = {
    0: "Strategy 0",
    1: "Strategy 1_1",
    2: "Strategy 2_1",
    3: "Strategy 3_1",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate combined all-model figures for text-embedding-3-large "
            "descriptor/Morgan data and Gemini descriptor data."
        )
    )
    parser.add_argument(
        "--score-column",
        choices=["semantic_similarity", "S_global", "S_coverage", "S_final"],
        default="semantic_similarity",
        help="Score column to aggregate.",
    )
    parser.add_argument(
        "--descriptor-details",
        type=Path,
        default=TEXT_DESCRIPTOR_DETAILS,
        help="text-embedding-3-large per-sample CSV for Strategy 0/1/2/3.",
    )
    parser.add_argument(
        "--morgan-details",
        type=Path,
        default=TEXT_MORGAN_DETAILS,
        help="text-embedding-3-large per-sample CSV for Strategy 0/1_1/2_1/3_1.",
    )
    parser.add_argument(
        "--gemini-descriptor-details",
        type=Path,
        default=GEMINI_DESCRIPTOR_DETAILS,
        help="gemini-embedding-2 per-sample CSV for Strategy 0/1/2/3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the two combined PNG files.",
    )
    return parser.parse_args()


def read_detail_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def prepare_detail_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["strategy_index"] = pd.to_numeric(df["strategy_index"], errors="coerce").astype("Int64")
    df["prediction_correct"] = (
        df["prediction_correct"].astype(str).str.lower().map({"true": True, "correct": True, "false": False, "wrong": False, "incorrect": False})
    )
    for score_column in [*SCORE_COLUMNS, "D_final", "alignment_zscore"]:
        if score_column in df.columns:
            df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    return df.dropna(subset=["strategy_index", "prediction_correct"])


def compute_similarity_metrics(details: pd.DataFrame) -> pd.DataFrame:
    """Recompute the final similarity instead of trusting the stored result."""
    required = {"S_global", "S_coverage"}
    missing = required - set(details.columns)
    if missing:
        raise ValueError(f"Cannot calculate S_final; missing columns: {sorted(missing)}")
    frame = details.copy()
    s_global = pd.to_numeric(frame["S_global"], errors="coerce")
    s_coverage = pd.to_numeric(frame["S_coverage"], errors="coerce")
    calculated = 0.3 * s_global + 0.7 * s_coverage
    if "S_final" in frame.columns:
        stored = pd.to_numeric(frame["S_final"], errors="coerce")
        comparable = stored.notna() & calculated.notna()
        if comparable.any() and not np.allclose(
            stored[comparable], calculated[comparable], rtol=1e-9, atol=1e-10
        ):
            raise ValueError("Stored S_final does not match 0.3*S_global + 0.7*S_coverage")
    frame["S_final"] = calculated
    frame["semantic_similarity"] = calculated
    return frame



def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_names = (
        ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    )
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def text_y_centered_on(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    center_y: float,
) -> float:
    bbox = draw.textbbox((0, 0), text, font=font)
    return center_y - (bbox[1] + bbox[3]) / 2


def hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5)) + (alpha,)


def figure_title(model_name: str) -> str:
    return f"OpenAI similarity to literature references ({display_model_title(model_name)})"


def fit_font_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    start_size: int,
    min_size: int,
    bold: bool = False,
) -> ImageFont.ImageFont:
    for size in range(start_size, min_size - 1, -2):
        font = load_font(size, bold=bold)
        text_w, _ = text_size(draw, text, font)
        if text_w <= max_width:
            return font
    return load_font(min_size, bold=bold)


def plot_summary(
    details: pd.DataFrame,
    output: Path,
    *,
    title: str,
    score_column: str,
    connect_matching_points: bool = False,
    custom_legend_layout: bool = True,
    strategy_labels: dict[int, str] = DESCRIPTOR_STRATEGY_LABELS,
    x_limits: tuple[float, float] = (0.33, 0.80),
    x_ticks: Optional[np.ndarray] = None,
    x_label: str = "Mean semantic similarity",
) -> None:
    use_custom_layout = connect_matching_points and custom_legend_layout
    width, height = (2100, 1050) if use_custom_layout else ((1500, 1320) if connect_matching_points else (1500, 1200))
    margin_left = 260
    margin_right = 540 if use_custom_layout else 120
    margin_top = 160
    margin_bottom = 210 if use_custom_layout else (420 if connect_matching_points else 300)
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    x_min, x_max = x_limits
    if x_ticks is None:
        x_ticks = np.arange(np.ceil(x_min * 10) / 10, x_max + 1e-9, 0.10)
    y_min, y_max = -0.5, 3.5

    def x_to_px(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value: float) -> float:
        return plot_top + (value - y_min) / (y_max - y_min) * plot_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_max_width = plot_width if use_custom_layout else width - 120
    title_font = fit_font_to_width(draw, title, max_width=title_max_width, start_size=58, min_size=34, bold=True)
    label_font = load_font(42 if use_custom_layout else 46)
    tick_font = load_font(42)
    legend_font = load_font(36 if use_custom_layout else 43)
    value_font = load_font(26)

    title_w, _ = text_size(draw, title, title_font)
    title_x = plot_left + (plot_width - title_w) / 2 if use_custom_layout else (width - title_w) / 2
    draw.text((title_x, 68), title, fill="#111111", font=title_font)

    legend_rows = [[
        {"kind": "color", "label": "Correct", "color": VIOLIN_LABEL_STYLE[True]["color"]},
        {"kind": "color", "label": "Wrong", "color": VIOLIN_LABEL_STYLE[False]["color"]},
        {"kind": "median_line", "label": "Mean", "color": "#111111"},
    ]]
    if connect_matching_points:
        legend_rows.append(
            [
                {"kind": "line", "label": "W \u2192 C", "color": INCORRECT_TO_CORRECT_LINE_COLOR},
                {"kind": "line", "label": "C \u2192 W", "color": CORRECT_TO_INCORRECT_LINE_COLOR},
                {"kind": "line", "label": "Unchanged", "color": SAME_CORRECTNESS_LINE_COLOR},
            ]
        )

    marker_r = 11
    legend_gap = 42
    legend_row_gap = 18
    legend_padding_x, legend_padding_y = 24, 15
    legend_row_layouts = []
    for legend_items in legend_rows:
        item_widths = []
        item_heights = []
        for item in legend_items:
            label = item["label"]
            text_w, text_h = text_size(draw, label, legend_font)
            symbol_width = 64 if item["kind"] in {"median_line", "line"} else marker_r * 2
            item_widths.append(symbol_width + 14 + text_w)
            item_heights.append(max(marker_r * 2, text_h))
        row_content_w = sum(item_widths) + legend_gap * (len(legend_items) - 1)
        row_h = max(item_heights)
        legend_row_layouts.append(
            {"items": legend_items, "item_widths": item_widths, "content_width": row_content_w, "height": row_h}
        )
    legend_content_w = max(row["content_width"] for row in legend_row_layouts)
    legend_content_h = sum(row["height"] for row in legend_row_layouts) + legend_row_gap * (len(legend_row_layouts) - 1)
    legend_w = legend_content_w + legend_padding_x * 2
    legend_h = legend_content_h + legend_padding_y * 2
    legend_x = plot_left + (plot_width - legend_w) / 2
    legend_y = plot_bottom + 165
    draw.rounded_rectangle(
        [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
        radius=5,
        fill="white",
        outline="#666666",
        width=2,
    )
    cursor_y = legend_y + legend_padding_y
    for row in legend_row_layouts:
        cursor_x = legend_x + (legend_w - row["content_width"]) / 2
        cy = cursor_y + row["height"] / 2
        for item, item_w in zip(row["items"], row["item_widths"]):
            label = item["label"]
            color = item["color"]
            if item["kind"] in {"median_line", "line"}:
                x0 = cursor_x
                x1 = cursor_x + 64
                draw.line([(x0, cy), (x1, cy)], fill=color, width=7 if item["kind"] == "median_line" else 5)
                text_x = cursor_x + 64 + 14
            else:
                marker_center_x = cursor_x + 32
                draw.ellipse(
                    [
                        marker_center_x - marker_r,
                        cy - marker_r,
                        marker_center_x + marker_r,
                        cy + marker_r,
                    ],
                    fill=color,
                    outline=color,
                )
                text_x = cursor_x + 64 + 14
            draw.text(
                (text_x, text_y_centered_on(draw, label, legend_font, cy)),
                label,
                fill="#111111",
                font=legend_font,
            )
            cursor_x += item_w + legend_gap
        cursor_y += row["height"] + legend_row_gap

    # Replace the compact bottom legend with a publication-style, sectioned
    # legend to the right of each connected model plot.
    if use_custom_layout:
        draw.rectangle([plot_right + 1, plot_top, width, plot_bottom], fill="white")
        draw.rectangle(
            [legend_x - 6, legend_y - 6, legend_x + legend_w + 6, legend_y + legend_h + 6],
            fill="white",
        )

        legend_font = load_font(30)
        section_title_font = load_font(30, bold=True)
        symbol_width = 54
        symbol_gap = 16
        row_gap = 14
        separator_gap = 18
        legend_padding_x, legend_padding_y = 24, 22
        legend_sections = [
            (
                "Prediction correctness",
                [
                    {"kind": "point", "label": "Correct", "color": VIOLIN_LABEL_STYLE[True]["color"]},
                    {"kind": "point", "label": "Wrong", "color": VIOLIN_LABEL_STYLE[False]["color"]},
                ],
            ),
            (
                "Distributions",
                [
                    {"kind": "mean_line", "label": "Mean", "color": "#111111"},
                    {
                        "kind": "distribution",
                        "label": "Correct distribution",
                        "color": VIOLIN_LABEL_STYLE[True]["color"],
                        "fill": "#fafcff",
                    },
                    {
                        "kind": "distribution",
                        "label": "Wrong distribution",
                        "color": VIOLIN_LABEL_STYLE[False]["color"],
                        "fill": "#fffaf7",
                    },
                ],
            ),
            (
                "Correctness transition",
                [
                    {"kind": "line", "label": "W \u2192 C", "color": INCORRECT_TO_CORRECT_LINE_COLOR},
                    {"kind": "line", "label": "C \u2192 W", "color": CORRECT_TO_INCORRECT_LINE_COLOR},
                    {"kind": "line", "label": "Unchanged", "color": SAME_CORRECTNESS_LINE_COLOR},
                ],
            ),
        ]

        legend_content_w = 0
        legend_content_h = 0
        for section_index, (section_title, section_items) in enumerate(legend_sections):
            title_w, title_h = text_size(draw, section_title, section_title_font)
            title_row_h = max(title_h, 24)
            legend_content_w = max(legend_content_w, title_w)
            legend_content_h += title_row_h + row_gap
            for item in section_items:
                text_w, text_h = text_size(draw, item["label"], legend_font)
                legend_content_w = max(legend_content_w, symbol_width + symbol_gap + text_w)
                legend_content_h += max(text_h, 24)
            legend_content_h += row_gap * (len(section_items) - 1)
            if section_index < len(legend_sections) - 1:
                legend_content_h += separator_gap * 2 + 1

        legend_w = legend_content_w + legend_padding_x * 2
        legend_h = legend_content_h + legend_padding_y * 2
        legend_x = plot_right + 45
        legend_y = plot_top + (plot_height - legend_h) / 2
        draw.rounded_rectangle(
            [legend_x, legend_y, legend_x + legend_w, legend_y + legend_h],
            radius=5,
            fill="white",
            outline="#666666",
            width=2,
        )

        def draw_custom_legend_symbol(item: dict[str, str], x: float, cy: float) -> None:
            kind = item["kind"]
            color = item["color"]
            if kind == "point":
                marker_center_x = x + symbol_width / 2
                draw.ellipse(
                    [
                        marker_center_x - marker_r,
                        cy - marker_r,
                        marker_center_x + marker_r,
                        cy + marker_r,
                    ],
                    fill=color,
                    outline=color,
                )
            elif kind == "distribution":
                draw.rectangle(
                    [x + 4, cy - 11, x + symbol_width - 4, cy + 11],
                    fill=item["fill"],
                    outline=color,
                    width=2,
                )
            else:
                draw.line(
                    [(x + 2, cy), (x + symbol_width - 2, cy)],
                    fill=color,
                    width=6 if kind == "mean_line" else 4,
                )

        cursor_y = legend_y + legend_padding_y
        for section_index, (section_title, section_items) in enumerate(legend_sections):
            _, title_h = text_size(draw, section_title, section_title_font)
            title_row_h = max(title_h, 24)
            title_cy = cursor_y + title_row_h / 2
            draw.text(
                (
                    legend_x + legend_padding_x,
                    text_y_centered_on(draw, section_title, section_title_font, title_cy),
                ),
                section_title,
                fill="#111111",
                font=section_title_font,
            )
            cursor_y += title_row_h + row_gap
            for item in section_items:
                _, text_h = text_size(draw, item["label"], legend_font)
                row_h = max(text_h, 24)
                cy = cursor_y + row_h / 2
                symbol_x = legend_x + legend_padding_x
                draw_custom_legend_symbol(item, symbol_x, cy)
                draw.text(
                    (
                        symbol_x + symbol_width + symbol_gap,
                        text_y_centered_on(draw, item["label"], legend_font, cy),
                    ),
                    item["label"],
                    fill="#111111",
                    font=legend_font,
                )
                cursor_y += row_h + row_gap
            cursor_y -= row_gap
            if section_index < len(legend_sections) - 1:
                cursor_y += separator_gap
                draw.line(
                    [(legend_x, cursor_y), (legend_x + legend_w, cursor_y)],
                    fill="#a8a8a8",
                    width=1,
                )
                cursor_y += separator_gap + 1

    band_gap = 10
    band_fill = "#ffffff"  #"#f7f7f7"
    for strategy_index in STRATEGIES:
        band_top = y_to_px(strategy_index - 0.5) + band_gap / 2
        band_bottom = y_to_px(strategy_index + 0.5) - band_gap / 2
        draw.rounded_rectangle(
            [plot_left, band_top, plot_right, band_bottom],
            radius=6,
            fill=band_fill,
        )

    for tick in x_ticks:
        x = x_to_px(float(tick))
        draw.line([(x, plot_top), (x, plot_bottom)], fill="#e6e3e3", width=2)
        label = f"{tick:.2f}"
        label_w, _ = text_size(draw, label, tick_font)
        draw.text((x - label_w / 2, plot_bottom + 22), label, fill="#222222", font=tick_font)

    for strategy_index in STRATEGIES:
        y = y_to_px(float(strategy_index))
        label = strategy_labels[strategy_index]
        label_w, label_h = text_size(draw, label, tick_font)
        draw.text(
            (plot_left - label_w - 24, y - label_h / 2),
            label,
            fill="#222222",
            font=tick_font,
        )

    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill="#333333", width=3)
    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill="#333333", width=3)

    x_label_w, _ = text_size(draw, x_label, label_font)
    draw.text(
        (plot_left + (plot_width - x_label_w) / 2, height - margin_bottom + 88),
        x_label,
        fill="#111111",
        font=label_font,
    )

    overlay_scale = 3
    overlay = Image.new("RGBA", (width * overlay_scale, height * overlay_scale), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    def scaled_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x * overlay_scale, y * overlay_scale) for x, y in points]

    def scaled_box(box: list[float]) -> list[float]:
        return [value * overlay_scale for value in box]

    violin_half_width = 48
    scatter_radius = 5
    bins = np.linspace(x_min, x_max, 90)
    centers = (bins[:-1] + bins[1:]) / 2
    kernel_positions = np.arange(-8, 9, dtype=float)
    smooth_kernel = np.exp(-0.5 * (kernel_positions / 3.0) ** 2)
    smooth_kernel = smooth_kernel / smooth_kernel.sum()
    violin_records = []

    for strategy_index in STRATEGIES:
        for prediction_correct in LABEL_ORDER:
            style = VIOLIN_LABEL_STYLE[prediction_correct]
            group = details[
                (details["strategy_index"].astype(int) == strategy_index)
                & (details["prediction_correct"] == prediction_correct)
            ]
            group = group.copy()
            group["_plot_score"] = pd.to_numeric(group[score_column], errors="coerce")
            group = group.dropna(subset=["_plot_score"])
            group = group[(group["_plot_score"] >= x_min) & (group["_plot_score"] <= x_max)]
            values = group["_plot_score"].to_numpy(dtype=float)
            if values.size == 0:
                continue

            y_center = y_to_px(strategy_index + style["y_offset"])
            hist, _ = np.histogram(values, bins=bins)
            density = np.convolve(hist.astype(float), smooth_kernel, mode="same")
            if density.max() == 0:
                density = np.ones_like(density)
            density = density / density.max() * violin_half_width
            upper = [(x_to_px(float(x_value)), y_center - float(width_value)) for x_value, width_value in zip(centers, density)]
            lower = [(x_to_px(float(x_value)), y_center + float(width_value)) for x_value, width_value in zip(centers[::-1], density[::-1])]

            rng_seed = int(hashlib.sha256(f"{title}|{strategy_index}|{prediction_correct}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)
            rng = np.random.default_rng(rng_seed)
            jitter = rng.uniform(-violin_half_width * 0.55, violin_half_width * 0.55, size=values.size)
            mean_value = float(np.mean(values))
            median_x = x_to_px(mean_value)
            median_half_width = float(np.interp(mean_value, centers, density))
            violin_records.append(
                {
                    "strategy_index": strategy_index,
                    "prediction_correct": prediction_correct,
                    "color": style["color"],
                    "polygon": upper + lower,
                    "points": [
                        {
                            "x": x_to_px(float(score_value)),
                            "y": y_center + float(jitter_value),
                            "source_model": str(source_model),
                            "molecule_id": str(molecule_id),
                            "experiment_id": str(experiment_id),
                            "strategy_index": strategy_index,
                            "prediction_correct": prediction_correct,
                        }
                        for score_value, source_model, molecule_id, experiment_id, jitter_value in zip(
                            group["_plot_score"],
                            group["source_model"],
                            group["molecule_id"],
                            group["experiment_id"],
                            jitter,
                        )
                    ],
                    "median_x": median_x,
                    "median_y": y_center,
                    "median_half_width": median_half_width,
                }
            )

    def records_for_label(prediction_correct: bool) -> list[dict[str, Any]]:
        return [
            record
            for record in sorted(violin_records, key=lambda item: item["strategy_index"])
            if record["prediction_correct"] == prediction_correct
        ]

    def draw_violin_records(records: list[dict[str, Any]]) -> None:
        for record in records:
            color = record["color"]
            rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
            fill_alpha = CONNECTED_VIOLIN_FILL_ALPHA if use_custom_layout else (0 if connect_matching_points else 70)
            fill_rgba = rgb + (fill_alpha,)
            outline_rgba = rgb + (255,)
            polygon = record["polygon"]
            overlay_draw.polygon(scaled_points(polygon), fill=fill_rgba)
            overlay_draw.line(
                scaled_points(polygon + [polygon[0]]),
                fill=outline_rgba,
                width=(4 if connect_matching_points else 3) * overlay_scale,
            )

    def draw_median_connectors(records: list[dict[str, Any]]) -> None:
        for start, end in zip(records[:-1], records[1:]):
            overlay_draw.line(
                scaled_points(
                    [
                        (start["median_x"], start["median_y"] + start["median_half_width"]),
                        (end["median_x"], end["median_y"] - end["median_half_width"]),
                    ]
                ),
                fill=(105, 105, 105, 190),
                width=2 * overlay_scale,
            )

    def draw_scatter_records(records: list[dict[str, Any]]) -> None:
        for record in records:
            color = record["color"]
            rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
            point_rgba = rgb + (190,)
            for point in record["points"]:
                px = point["x"]
                py = point["y"]
                overlay_draw.ellipse(
                    scaled_box([px - scatter_radius, py - scatter_radius, px + scatter_radius, py + scatter_radius]),
                    fill=point_rgba,
                    outline=None,
                )

    def draw_matching_point_connectors(records: list[dict[str, Any]]) -> None:
        points_by_key: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
        for record in records:
            for point in record["points"]:
                key = (point["source_model"], point["molecule_id"], point["experiment_id"])
                points_by_key.setdefault(key, {})[point["strategy_index"]] = point
        same_segments = []
        incorrect_to_correct_segments = []
        correct_to_incorrect_segments = []
        for points_by_strategy in points_by_key.values():
            connected_points = [
                points_by_strategy[strategy_index]
                for strategy_index in STRATEGIES
                if strategy_index in points_by_strategy
            ]
            for start, end in zip(connected_points[:-1], connected_points[1:]):
                segment = [(start["x"], start["y"]), (end["x"], end["y"])]
                if (not start["prediction_correct"]) and end["prediction_correct"]:
                    incorrect_to_correct_segments.append(segment)
                elif start["prediction_correct"] and (not end["prediction_correct"]):
                    correct_to_incorrect_segments.append(segment)
                else:
                    same_segments.append(segment)

        for segments, line_color in [
            (same_segments, hex_to_rgba(SAME_CORRECTNESS_LINE_COLOR, SAME_CORRECTNESS_LINE_ALPHA)),
            (
                incorrect_to_correct_segments,
                hex_to_rgba(INCORRECT_TO_CORRECT_LINE_COLOR, INCORRECT_TO_CORRECT_LINE_ALPHA),
            ),
            (
                correct_to_incorrect_segments,
                hex_to_rgba(CORRECT_TO_INCORRECT_LINE_COLOR, CORRECT_TO_INCORRECT_LINE_ALPHA),
            ),
        ]:
            for segment in segments:
                overlay_draw.line(
                    scaled_points(segment),
                    fill=line_color,
                    width=1 * overlay_scale,
                )

    def draw_median_records(records: list[dict[str, Any]]) -> None:
        for record in records:
            median_x = record["median_x"]
            median_y = record["median_y"]
            median_half_width = record["median_half_width"]
            overlay_draw.line(
                scaled_points([(median_x, median_y - median_half_width), (median_x, median_y + median_half_width)]),
                fill=(0, 0, 0, 255),
                width=5 * overlay_scale,
            )

    true_records = records_for_label(True)
    false_records = records_for_label(False)

    if connect_matching_points:
        draw_violin_records(true_records)
        draw_median_records(true_records)
        draw_violin_records(false_records)
        draw_matching_point_connectors(true_records + false_records)
        draw_scatter_records(true_records)
        draw_scatter_records(false_records)
        draw_median_records(false_records)
    else:
        draw_violin_records(true_records)
        draw_median_connectors(true_records)
        draw_median_records(true_records)
        draw_scatter_records(true_records)
        draw_violin_records(false_records)
        draw_median_connectors(false_records)
        draw_scatter_records(false_records)
        draw_median_records(false_records)

    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    overlay = overlay.resize((width, height), resample=resample_filter)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    image.save(output, dpi=(300, 300))



def display_model_title(model_name: str) -> str:
    labels = {
        "grok-4.3": "Grok 4.3",
        "deepseek-v3.2-think": "DeepSeek-V3.2",
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
        "gpt-4o": "GPT-4o",
        "llama-4-scout": "Llama 4 Scout",
    }
    return labels.get(model_name, model_name[:1].upper() + model_name[1:] if model_name else model_name)


def draw_combined_legend(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y: float,
    font: ImageFont.ImageFont,
) -> None:
    items = [
        {"kind": "color", "label": "Correct", "color": VIOLIN_LABEL_STYLE[True]["color"]},
        {"kind": "color", "label": "Wrong", "color": VIOLIN_LABEL_STYLE[False]["color"]},
        {"kind": "mean_line", "label": "Mean", "color": "#111111"},
        {"kind": "line", "label": "W \u2192 C", "color": INCORRECT_TO_CORRECT_LINE_COLOR},
        {"kind": "line", "label": "C \u2192 W", "color": CORRECT_TO_INCORRECT_LINE_COLOR},
        {"kind": "line", "label": "Unchanged", "color": SAME_CORRECTNESS_LINE_COLOR},
    ]
    marker_r = 8
    symbol_width = 48
    symbol_gap = 12
    item_gap = 34
    padding_x, padding_y = 22, 14
    item_widths = []
    item_heights = []
    for item in items:
        text_w, text_h = text_size(draw, item["label"], font)
        symbol_w = symbol_width if item["kind"] in {"mean_line", "line"} else marker_r * 2
        item_widths.append(symbol_w + symbol_gap + text_w)
        item_heights.append(max(marker_r * 2, text_h))
    content_w = sum(item_widths) + item_gap * (len(items) - 1)
    content_h = max(item_heights)
    legend_w = content_w + padding_x * 2
    legend_h = content_h + padding_y * 2
    legend_x = x - legend_w / 2
    draw.rounded_rectangle(
        [legend_x, y, legend_x + legend_w, y + legend_h],
        radius=5,
        fill="white",
        outline="#666666",
        width=2,
    )
    cursor_x = legend_x + padding_x
    cy = y + legend_h / 2
    for item, item_w in zip(items, item_widths):
        color = item["color"]
        if item["kind"] in {"mean_line", "line"}:
            draw.line(
                [(cursor_x, cy), (cursor_x + symbol_width, cy)],
                fill=color,
                width=5 if item["kind"] == "mean_line" else 4,
            )
            text_x = cursor_x + symbol_width + symbol_gap
        else:
            draw.ellipse(
                [cursor_x, cy - marker_r, cursor_x + marker_r * 2, cy + marker_r],
                fill=color,
                outline=color,
            )
            text_x = cursor_x + marker_r * 2 + symbol_gap
        draw.text(
            (text_x, text_y_centered_on(draw, item["label"], font, cy)),
            item["label"],
            fill="#111111",
            font=font,
        )
        cursor_x += item_w + item_gap


def build_combined_figure(model_panels: dict[str, Path], output: Path) -> None:
    row_models = [MODEL_ORDER[:2], MODEL_ORDER[2:]]
    missing = [model for model in MODEL_ORDER if model not in model_panels]
    if missing:
        raise ValueError(f"Missing model panels: {', '.join(missing)}")

    source_panel_width, source_panel_height = 1500, 1040
    source_plot_center_x = (260 + 1380) / 2
    first_crop = (0, 0, 1420, source_panel_height)
    inner_crop = (220, 0, 1420, source_panel_height)
    top_source_height = 965
    bottom_panel_height = 620
    panel_scale = bottom_panel_height / source_panel_height
    top_panel_height = round(top_source_height * panel_scale)
    first_panel_width = round((first_crop[2] - first_crop[0]) * panel_scale)
    inner_panel_width = round((inner_crop[2] - inner_crop[0]) * panel_scale)
    row_heights = [top_panel_height, bottom_panel_height]
    gap_x, gap_y = 10, 18
    margin_x, margin_top = 52, 42
    legend_area_h = 120
    row_widths = [
        first_panel_width + gap_x + inner_panel_width,
        first_panel_width + gap_x + inner_panel_width + gap_x + inner_panel_width,
    ]
    canvas_width = margin_x * 2 + max(row_widths)
    canvas_height = margin_top + sum(row_heights) + gap_y + legend_area_h
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS

    for row_index, models in enumerate(row_models):
        panel_widths = [
            first_panel_width if column_index == 0 else inner_panel_width
            for column_index in range(len(models))
        ]
        row_width = sum(panel_widths) + gap_x * (len(models) - 1)
        cursor_x = (canvas_width - row_width) // 2
        y = margin_top if row_index == 0 else margin_top + row_heights[0] + gap_y
        for column_index, model in enumerate(models):
            with Image.open(model_panels[model]) as source_image:
                panel = source_image.convert("RGB").crop((0, 0, source_panel_width, source_panel_height))
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rectangle([0, 0, source_panel_width, 150], fill="white")
            title = display_model_title(model)
            title_font = fit_font_to_width(
                panel_draw,
                title,
                max_width=source_panel_width - 160,
                start_size=48,
                min_size=32,
                bold=True,
            )
            title_w, _ = text_size(panel_draw, title, title_font)
            panel_draw.text(
                (source_plot_center_x - title_w / 2, 72),
                title,
                fill="#111111",
                font=title_font,
            )
            if column_index > 0:
                panel_draw.rectangle([0, 145, 264, 910], fill="white")
                crop_box = inner_crop
                target_width = inner_panel_width
            else:
                crop_box = first_crop
                target_width = first_panel_width
            crop_bottom = top_source_height if row_index == 0 else source_panel_height
            panel = panel.crop((crop_box[0], crop_box[1], crop_box[2], crop_bottom)).resize(
                (target_width, row_heights[row_index]),
                resample=resample_filter,
            )
            canvas.paste(panel, (cursor_x, y))
            cursor_x += target_width + gap_x

    draw_combined_legend(
        ImageDraw.Draw(canvas),
        x=canvas_width / 2,
        y=canvas_height - legend_area_h + 24,
        font=load_font(34),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image_to_replace = output.resolve()
    if image_to_replace.exists():
        image_to_replace.unlink()
    canvas.save(image_to_replace, dpi=(300, 300))


def validate_dataset(details: pd.DataFrame, source: Path) -> None:
    models = set(details["source_model"].dropna().astype(str))
    missing_models = set(MODEL_ORDER) - models
    if missing_models:
        raise ValueError(f"{source} is missing models: {sorted(missing_models)}")
    for model in MODEL_ORDER:
        strategy_indexes = set(
            details.loc[details["source_model"].astype(str) == model, "strategy_index"].astype(int)
        )
        if strategy_indexes != set(STRATEGIES):
            raise ValueError(f"{source} has incomplete strategies for {model}: {strategy_indexes}")


def render_combined_dataset(
    details_path: Path,
    output: Path,
    *,
    score_column: str,
    strategy_labels: dict[int, str],
    minimum_x: Optional[float] = None,
    minimum_upper: Optional[float] = 0.80,
    tick_step: float = 0.10,
    x_label: str = "Mean semantic similarity",
    frame_transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> None:
    if not details_path.is_file():
        raise FileNotFoundError(f"Detail CSV not found: {details_path}")
    details = prepare_detail_frame(read_detail_rows(details_path))
    if frame_transform is not None:
        details = frame_transform(details)
    validate_dataset(details, details_path)
    score_values = pd.to_numeric(details[score_column], errors="coerce").dropna().to_numpy(float)
    value_span = max(float(np.ptp(score_values)), 0.1)
    padding = max(0.01, value_span * 0.04)
    automatic_lower = float(np.floor((score_values.min() - padding) * 10) / 10)
    if minimum_upper is not None:
        automatic_lower = min(0.33, automatic_lower)
    lower = minimum_x if minimum_x is not None else automatic_lower
    automatic_upper = float(np.ceil((score_values.max() + padding) * 10) / 10)
    upper = max(minimum_upper, automatic_upper) if minimum_upper is not None else automatic_upper
    if minimum_x is None and minimum_upper == 0.80 and score_values.min() >= 0.33 and score_values.max() <= 0.80:
        lower, upper = 0.33, 0.80
    x_ticks = np.arange(np.ceil(lower / tick_step) * tick_step, upper + 1e-9, tick_step)
    with TemporaryDirectory(prefix=".combined-panels-", dir=output.parent) as temporary_dir:
        model_panels = {}
        for model in MODEL_ORDER:
            model_frame = details[details["source_model"].astype(str) == model]
            panel_path = Path(temporary_dir) / f"{model}.png"
            plot_summary(
                model_frame,
                panel_path,
                title=figure_title(model),
                score_column=score_column,
                connect_matching_points=True,
                custom_legend_layout=False,
                strategy_labels=strategy_labels,
                x_limits=(lower, upper),
                x_ticks=x_ticks,
                x_label=x_label,
            )
            model_panels[model] = panel_path
        build_combined_figure(model_panels, output)


SIMILARITY_COLUMNS: list[str] = [
    "strategy", "strategy_index", "prediction_correct",
    "prediction_correct_label", "molecule_id", "experiment_id",
    "solvent_additive", "semantic_similarity", "S_global", "S_coverage",
    "S_final", "source_model", "chemical_description", "expected_result",
    "selected_round_count", "selected_rounds", "round_similarity_values",
    "round_S_global_values", "round_S_coverage_values", "round_S_final_values",
]

SEMCSE_COLUMNS = [
    "strategy", "strategy_index", "prediction_correct",
    "prediction_correct_label", "molecule_id", "experiment_id",
    "solvent_additive", "source_model", "chemical_description",
    "expected_result", "D_global", "D_coverage",
    "matched_distance_recomputed", "null_mean_distance",
    "null_std_distance", "D_final", "alignment_zscore",
]


@dataclass(frozen=True)
class CachedDocument:
    # Keep plotting importable without PyTorch; concrete tensors are validated
    # only on the missing-CSV cache-rebuild path.
    document: Any
    chunks: Any
    weights: Any


def _load_module(filename: str, name: str) -> Any:
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local cache module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LocalCacheReader:
    def __init__(self, backend: str, cache_root: Path = CACHE_ROOT) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required only when rebuilding a missing CSV from caches."
            )
        self.torch = torch
        configurations = {
            "text": (
                "precompute_text_embedding_3_large_reference_embeddings.py",
                "text-embedding-3-large",
            ),
            "gemini": (
                "precompute_gemini_embedding_2_reference_embeddings.py",
                "gemini-embedding-2",
            ),
            "semcse": (
                "precompute_semcse_reference_embeddings.py",
                "CLAUSE-Bielefeld/SemCSE",
            ),
        }
        if backend not in configurations:
            raise ValueError(f"Unknown cache backend: {backend}")
        filename, model = configurations[backend]
        module = _load_module(filename, f"qla_local_cache_{backend}")
        module.EMBEDDING_CACHE_DIR = Path(cache_root)
        self.keyer = module.CacheKeyOnlyEmbedder(model)
        self.backend = backend
        self.memory: dict[str, CachedDocument] = {}

    @property
    def cache_dir(self) -> Path:
        return self.keyer.document_embedding_cache_dir

    def load(self, text: str) -> CachedDocument:
        cleaned = str(text).strip()
        key = self.keyer.cache_key(cleaned)
        cached = self.memory.get(key)
        if cached is not None:
            return cached
        path = self.cache_dir / f"{key}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Required local cache is missing: {path}\n"
                "Run the matching precompute_*_reference_embeddings.py script "
                "from this folder first."
            )
        try:
            payload = self.torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = self.torch.load(path, map_location="cpu")
        chunks = payload["chunk_embeddings"].float().cpu()
        if chunks.ndim == 1:
            chunks = chunks.unsqueeze(0)
        weights = payload["chunk_weights"]
        if not isinstance(weights, self.torch.Tensor):
            weights = self.torch.tensor(weights, dtype=self.torch.float32)
        document = CachedDocument(
            document=payload["document_embedding"].float().cpu(),
            chunks=chunks,
            weights=weights.float().cpu(),
        )
        self.memory[key] = document
        return document

    def score(self, left: str, right: str) -> tuple[float, float, float]:
        a = self.load(left)
        b = self.load(right)
        if self.backend in {"text", "gemini"}:
            global_value = float(self.torch.dot(a.document, b.document).item())
            matrix = a.chunks @ b.chunks.T
            left_best = matrix.max(dim=1).values
            right_best = matrix.max(dim=0).values
        else:
            global_value = float(self.torch.dist(a.document, b.document, p=2).item())
            matrix = self.torch.cdist(a.chunks, b.chunks, p=2)
            left_best = matrix.min(dim=1).values
            right_best = matrix.min(dim=0).values
        left_coverage = float((left_best * a.weights).sum() / a.weights.sum())
        right_coverage = float((right_best * b.weights).sum() / b.weights.sum())
        coverage = (left_coverage + right_coverage) / 2.0
        final = 0.3 * global_value + 0.7 * coverage
        return global_value, coverage, final


def _records_by_group(
    source_table: Path,
    strategies: set[str],
) -> dict[tuple[str, str, int, int], list[CompactSourceRecord]]:
    groups: dict[tuple[str, str, int, int], list[CompactSourceRecord]] = defaultdict(list)
    for record in iter_source_records(
        source_table,
        source_models=set(MODEL_ORDER),
        strategies=strategies,
    ):
        if not record.mechanistic_explanation or not record.reference_text:
            continue
        groups[
            (record.source_model, record.strategy, record.molecule_id, record.experiment_id)
        ].append(record)
    for key, records in groups.items():
        records.sort(key=lambda record: record.round_number)
        if [record.round_number for record in records] != [1, 2, 3]:
            raise ValueError(f"Expected rounds 1/2/3 for {key}")
    expected = len(MODEL_ORDER) * len(strategies) * 130
    if len(groups) != expected:
        raise ValueError(f"Expected {expected} complete sample groups, found {len(groups)}")
    return groups


def _selected_rounds(
    records: list[CompactSourceRecord],
) -> tuple[bool, list[CompactSourceRecord]]:
    correctness: list[bool] = []
    for record in records:
        predicted = predicted_result(record.prediction_json)
        expected = normalize_result(record.expected_result)
        if predicted is None or expected is None:
            raise ValueError(
                f"Cannot resolve prediction/correct answer for {record.source_model}, "
                f"{record.strategy}, molecule {record.molecule_id}, "
                f"experiment {record.experiment_id}, round {record.round_number}"
            )
        correctness.append(predicted == expected)
    majority_correct = sum(correctness) >= 2
    selected = [
        record for record, is_correct in zip(records, correctness)
        if is_correct is majority_correct
    ]
    return majority_correct, selected


def _strategy_name(code: str) -> str:
    return "Strategy " + code.removeprefix("S")


def build_similarity_rows(
    backend: str,
    strategies: Iterable[str],
    *,
    source_table: Path = DEFAULT_SOURCE_TABLE,
    cache_root: Path = CACHE_ROOT,
) -> pd.DataFrame:
    strategy_list = list(strategies)
    groups = _records_by_group(Path(source_table), set(strategy_list))
    reader = LocalCacheReader(backend, cache_root)
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for molecule_id, experiment_id in sorted(
            {(key[2], key[3]) for key in groups if key[0] == model}
        ):
            for index, strategy in enumerate(strategy_list):
                records = groups[(model, strategy, molecule_id, experiment_id)]
                majority_correct, selected = _selected_rounds(records)
                scores = [
                    reader.score(record.output_text, record.reference_text)
                    for record in selected
                ]
                first = records[0]
                global_values = [score[0] for score in scores]
                coverage_values = [score[1] for score in scores]
                final_values = [score[2] for score in scores]
                label = "correct" if majority_correct else "wrong"
                rows.append({
                    "strategy": _strategy_name(strategy),
                    "strategy_index": index,
                    "prediction_correct": label,
                    "prediction_correct_label": label,
                    "molecule_id": molecule_id,
                    "experiment_id": experiment_id,
                    "solvent_additive": first.solvent_additive,
                    "semantic_similarity": float(np.mean(final_values)),
                    "S_global": float(np.mean(global_values)),
                    "S_coverage": float(np.mean(coverage_values)),
                    "S_final": float(np.mean(final_values)),
                    "source_model": model,
                    "chemical_description": first.chemical_description,
                    "expected_result": first.expected_result,
                    "selected_round_count": len(selected),
                    "selected_rounds": "; ".join(
                        f"Round {record.round_number}" for record in selected
                    ),
                    "round_similarity_values": json.dumps(final_values),
                    "round_S_global_values": json.dumps(global_values),
                    "round_S_coverage_values": json.dumps(coverage_values),
                    "round_S_final_values": json.dumps(final_values),
                })
    frame = pd.DataFrame(rows)
    if len(frame) != 2600:
        raise ValueError(f"Expected 2600 similarity rows, found {len(frame)}")
    return frame[SIMILARITY_COLUMNS]


def build_semcse_alignment_rows(
    *,
    source_table: Path = DEFAULT_SOURCE_TABLE,
    cache_root: Path = CACHE_ROOT,
) -> pd.DataFrame:
    strategies = ["S0", "S1", "S2", "S3"]
    groups = _records_by_group(Path(source_table), set(strategies))
    reader = LocalCacheReader("semcse", cache_root)
    sample_records: dict[tuple[int, int], CompactSourceRecord] = {}
    for records in groups.values():
        first = records[0]
        sample_records[(first.molecule_id, first.experiment_id)] = first
    sample_keys = sorted(sample_records)
    reference_texts = [sample_records[key].reference_text for key in sample_keys]
    reference_index = {key: index for index, key in enumerate(sample_keys)}
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for molecule_id, experiment_id in sample_keys:
            for strategy_index, strategy in enumerate(strategies):
                records = groups[(model, strategy, molecule_id, experiment_id)]
                majority_correct, selected = _selected_rounds(records)
                round_vectors: list[list[tuple[float, float, float]]] = []
                for record in selected:
                    round_vectors.append([
                        reader.score(record.output_text, reference)
                        for reference in reference_texts
                    ])
                values = np.asarray(round_vectors, dtype=float)
                mean_global = values[:, :, 0].mean(axis=0)
                mean_coverage = values[:, :, 1].mean(axis=0)
                mean_final = values[:, :, 2].mean(axis=0)
                matched = reference_index[(molecule_id, experiment_id)]
                null = np.delete(mean_final, matched)
                null_mean = float(null.mean())
                null_std = float(null.std())
                first = records[0]
                rows.append({
                    "strategy": _strategy_name(strategy),
                    "strategy_index": strategy_index,
                    "prediction_correct": majority_correct,
                    "prediction_correct_label": "true" if majority_correct else "false",
                    "molecule_id": molecule_id,
                    "experiment_id": experiment_id,
                    "solvent_additive": first.solvent_additive,
                    "source_model": model,
                    "chemical_description": first.chemical_description,
                    "expected_result": first.expected_result,
                    "D_global": float(mean_global[matched]),
                    "D_coverage": float(mean_coverage[matched]),
                    "matched_distance_recomputed": float(mean_final[matched]),
                    "null_mean_distance": null_mean,
                    "null_std_distance": null_std,
                    "D_final": float(mean_final[matched]),
                    "alignment_zscore": (
                        (null_mean - float(mean_final[matched])) / null_std
                        if null_std > 0 else float("nan")
                    ),
                })
    frame = pd.DataFrame(rows)
    if len(frame) != 2600:
        raise ValueError(f"Expected 2600 SemCSE rows, found {len(frame)}")
    return frame[SEMCSE_COLUMNS]


DATASETS = {
    "text_descriptor": {
        "path": DATA_DIR / "text-embedding-3-large_physicochemical_descriptor_similarity_rows.csv",
        "cache": CACHE_ROOT / "text-embedding-3-large" / "documents",
        "precompute": "precompute_text_embedding_3_large_reference_embeddings.py",
    },
    "text_morgan": {
        "path": DATA_DIR / "text-embedding-3-large_morgan_fingerprint_similarity_rows.csv",
        "cache": CACHE_ROOT / "text-embedding-3-large" / "documents",
        "precompute": "precompute_text_embedding_3_large_reference_embeddings.py",
    },
    "gemini_descriptor": {
        "path": DATA_DIR / "gemini-embedding-2_physicochemical_descriptor_similarity_rows.csv",
        "cache": CACHE_ROOT / "gemini-embedding-2" / "documents",
        "precompute": "precompute_gemini_embedding_2_reference_embeddings.py",
    },
    "semcse_descriptor": {
        "path": DATA_DIR / "semcse_physicochemical_descriptor_alignment_zscore_D_final_rows.csv",
        "cache": CACHE_ROOT / "CLAUSE-Bielefeld_SemCSE" / "documents",
        "precompute": "precompute_semcse_reference_embeddings.py",
    },
}


def dataset_key_for_path(path: Path) -> str:
    resolved = path.resolve()
    for key, config in DATASETS.items():
        if resolved == Path(config["path"]).resolve():
            return key
    raise ValueError(f"No local-cache fallback is configured for CSV: {path}")


def _require_local_inputs(key: str) -> None:
    config = DATASETS[key]
    cache_dir = Path(config["cache"])
    if not DEFAULT_SOURCE_TABLE.is_file():
        raise FileNotFoundError(
            f"Required compressed source table is missing: {DEFAULT_SOURCE_TABLE}"
        )
    if not cache_dir.is_dir() or next(cache_dir.glob("*.pt"), None) is None:
        raise FileNotFoundError(
            f"Required CSV is missing: {config['path']}\n"
            f"Folder-local embedding cache is missing or empty: {cache_dir}\n"
            f"Generate it from the bundled source table with:\n"
            f"    python {config['precompute']}"
        )


def _build(key: str, destination: Path) -> None:
    # Keep the ordinary CSV-only plotting path lightweight. PyTorch and the
    # embedding backend are imported only when a missing CSV must be rebuilt.
    if key == "text_descriptor":
        frame = build_similarity_rows("text", ["S0", "S1", "S2", "S3"])
    elif key == "text_morgan":
        frame = build_similarity_rows("text", ["S0", "S1_1", "S2_1", "S3_1"])
    elif key == "gemini_descriptor":
        frame = build_similarity_rows("gemini", ["S0", "S1", "S2", "S3"])
    elif key == "semcse_descriptor":
        frame = build_semcse_alignment_rows()
    else:
        raise ValueError(f"Unknown dataset key: {key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")


def ensure_dataset_csv(key_or_path: str | Path) -> Path:
    """Return an existing CSV, or rebuild it without leaving this folder."""
    key = (
        dataset_key_for_path(Path(key_or_path))
        if isinstance(key_or_path, Path) or str(key_or_path) not in DATASETS
        else str(key_or_path)
    )
    destination = Path(DATASETS[key]["path"])
    if destination.is_file():
        return destination
    _require_local_inputs(key)
    print(f"CSV missing; rebuilding only from local source_data/cache: {destination}")
    _build(key, destination)
    if not destination.is_file():
        raise RuntimeError(f"Local cache reconstruction did not create: {destination}")
    return destination



def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        {
            "details_path": args.descriptor_details.resolve(),
            "output": output_dir / f"{TEXT_EMBEDDING_PREFIX}_strategy_similarity_all_models_physicochemical-descriptor.png",
            "strategy_labels": DESCRIPTOR_STRATEGY_LABELS,
            "score_column": args.score_column,
            "frame_transform": compute_similarity_metrics,
        },
        {
            "details_path": args.morgan_details.resolve(),
            "output": output_dir / f"{TEXT_EMBEDDING_PREFIX}_strategy_similarity_all_models_morgan-fingerprint.png",
            "strategy_labels": MORGAN_STRATEGY_LABELS,
            "score_column": args.score_column,
            "frame_transform": compute_similarity_metrics,
        },
        {
            "details_path": args.gemini_descriptor_details.resolve(),
            "output": output_dir / f"{GEMINI_EMBEDDING_PREFIX}_strategy_similarity_all_models_physicochemical-descriptor.png",
            "strategy_labels": DESCRIPTOR_STRATEGY_LABELS,
            "score_column": args.score_column,
            "minimum_x": 0.60,
            "frame_transform": compute_similarity_metrics,
        },
    ]
    for settings in outputs:
        settings["details_path"] = ensure_dataset_csv(settings["details_path"])
        render_combined_dataset(**settings)
        print(f"Wrote combined figure: {settings['output']}")


if __name__ == "__main__":
    main()






