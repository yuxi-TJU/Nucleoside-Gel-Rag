# RAG Reference-Molecule Overlap Analysis

This analysis compares the reference molecules selected by six RAG strategies:

- `S1`, `S2`, and `S3`: 24-dimensional descriptor distance;
- `S1_1`, `S2_1`, and `S3_1`: Morgan-fingerprint Tanimoto similarity;
- `S2`, `S3`, `S2_1`, and `S3_1` additionally use experimental-condition filtering.

Reference-set overlap is measured with the Jaccard index:

```text
Jaccard = intersection size / union size
```

The default retrieval depths are `max_k = 3, 5, 6`.

## Files

Inputs:

- `molecule_database.json`: molecules and experimental conditions;
- `molecule_descriptor_values_24d.csv`: molecule metadata and 24 descriptor columns;
- `literature_doi_by_molecule.csv`: molecule-to-DOI mapping.

Scripts:

- `build_reference_overlap_summary.py`: pairwise reference-set overlap;
- `build_combined_overlap_summary.py`: overlap and same-literature statistics.

Main outputs:

- `reference_overlap_summary_by_k.csv`;
- `combined_overlap_summary.csv`.

The reference-overlap script also writes `reference_molecule_sets_detailed.csv` and `reference_overlap_pairwise.csv`.

## Requirements

```powershell
pip install numpy pandas rdkit
```

## Run

From the project root:

```powershell
python overlap_summary_bundle/build_reference_overlap_summary.py
python overlap_summary_bundle/build_combined_overlap_summary.py
```

Existing output files are overwritten.

Common options:

```text
--max-k 3 5 6
--indices 1 2 3
--start-index 1 --end-index 20
--experiment-indices 1 2
--output-dir PATH
```

Use `python SCRIPT_NAME.py --help` for all options.

## Output columns

`reference_overlap_summary_by_k.csv`:

- `comparison_type`: strategy-pair identifier;
- `comparison_columns`: strategy-pair description;
- `max_k`: retrieval depth;
- `mean_jaccard`: mean Jaccard index.

`combined_overlap_summary.csv`:

- `metric_type`: `pairwise_molecule_overlap` or `strategy_same_literature_ratio`;
- `max_k`: retrieval depth;
- `strategy_explanation`: strategy description for same-literature rows;
- `comparison_type`, `comparison_columns`: strategy-pair information;
- `mean_jaccard_overlap`: mean Jaccard index;
- `mean_intersection_size`, `mean_union_size`: mean set sizes;
- `global_same_literature_reference_ratio`: same-literature reference slots divided by all reference slots.
