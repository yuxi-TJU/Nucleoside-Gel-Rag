from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_combined_strategy_similarity import (
    DEFAULT_OUTPUT_DIR,
    DESCRIPTOR_STRATEGY_LABELS,
    SCRIPT_DIR,
    ensure_dataset_csv,
    render_combined_dataset,
)


DEFAULT_DETAILS = (
    SCRIPT_DIR
    / "data"
    / "semcse_physicochemical_descriptor_alignment_zscore_D_final_rows.csv"
)
DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_DIR
    / "semcse_strategy_alignment_all_models_physicochemical-descriptor_alignment-zscore.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate D_final and alignment_zscore from the bundled SemCSE "
            "descriptor CSV, then generate one all-model alignment figure."
        )
    )
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def calculate_semcse_metrics(details: pd.DataFrame) -> pd.DataFrame:
    """Calculate the two retained SemCSE metrics from their component values."""
    required = {
        "D_global",
        "D_coverage",
        "matched_distance_recomputed",
        "null_mean_distance",
        "null_std_distance",
    }
    missing = required - set(details.columns)
    if missing:
        raise ValueError(
            f"Cannot calculate SemCSE metrics; missing columns: {sorted(missing)}"
        )

    frame = details.copy()
    d_global = pd.to_numeric(frame["D_global"], errors="coerce")
    d_coverage = pd.to_numeric(frame["D_coverage"], errors="coerce")
    matched_distance = pd.to_numeric(
        frame["matched_distance_recomputed"], errors="coerce"
    )
    null_mean = pd.to_numeric(frame["null_mean_distance"], errors="coerce")
    null_std = pd.to_numeric(frame["null_std_distance"], errors="coerce")

    calculated_d_final = 0.3 * d_global + 0.7 * d_coverage
    # The null-baseline stage compares every prediction with all candidate
    # references, so its matched distance is recalculated in that same matrix.
    calculated_zscore = (null_mean - matched_distance) / null_std.where(null_std > 0)

    for column, calculated in [
        ("D_final", calculated_d_final),
        ("alignment_zscore", calculated_zscore),
    ]:
        if column not in frame.columns:
            continue
        stored = pd.to_numeric(frame[column], errors="coerce")
        comparable = stored.notna() & calculated.notna()
        if comparable.any() and not np.allclose(
            stored[comparable],
            calculated[comparable],
            rtol=1e-9,
            atol=1e-10,
        ):
            max_difference = float(
                np.max(np.abs(stored[comparable] - calculated[comparable]))
            )
            raise ValueError(
                f"Stored {column} does not match its formula; "
                f"maximum absolute difference: {max_difference}"
            )

    frame["D_final"] = calculated_d_final
    frame["alignment_zscore"] = calculated_zscore
    return frame


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    details = ensure_dataset_csv(args.details.resolve())
    render_combined_dataset(
        details,
        output,
        score_column="alignment_zscore",
        strategy_labels=DESCRIPTOR_STRATEGY_LABELS,
        minimum_upper=None,
        tick_step=1.0,
        x_label="Mean alignment z-score",
        frame_transform=calculate_semcse_metrics,
    )
    print(f"Wrote SemCSE alignment figure: {output}")


if __name__ == "__main__":
    main()
