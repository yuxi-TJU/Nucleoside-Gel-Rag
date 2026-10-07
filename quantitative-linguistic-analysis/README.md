# Quantitative Linguistic Analysis

This folder is a self-contained release of the Figure 3a-d quantitative
linguistic analyses.

## Fast reproduction from bundled CSV files

The checked-in row-level CSV files are sufficient to regenerate every figure.
No API key, embedding model, or embedding cache is needed for this path.

```bash
cd quantitative-linguistic-analysis/3a_similarity_distributions
python -m pip install -r requirements.txt
python plot_combined_strategy_similarity.py
python plot_semcse_alignment.py

cd ..
python 3b_mean_rationale_mechanism_similarity/plot_strategy_outcome_similarity.py
python 3c_transition_group_mean_trajectories/plot_transition_group_mean_trajectories.py
python 3d_mean_local_mechanistic_discrimination/plot_local_mechanistic_discrimination.py
```

Filename meanings:

- `physicochemical-descriptor`: S0, S1, S2, S3.
- `morgan-fingerprint`: S0, S1_1, S2_1, S3_1.

## Bundled compressed source data

The original files and DB reference fields are consolidated in:

`3a_similarity_distributions/source_data/max0_and_max3_descriptor_morgan_source_records.csv.gz`

It contains 23,310 rows: five source models, three rounds, the shared Max-0/S0
baseline, Max-3 descriptor S1/S2/S3, and Max-3 Morgan S1_1/S2_1/S3_1. Each row
contains the exact output text, prediction JSON, identifiers, expected result,
and reference mechanism text. Its manifest records the schema and SHA-256.

## Optional: regenerate embedding caches from source_data

Caches are written only under:

`3a_similarity_distributions/embedding_cache/<embedding-model>/`

The scripts expose complete scoring methods as well as encoding:

- OpenAI/Gemini: `similarity(left, right)` returns `S_global`, `S_coverage`,
  and `S_final`.
- SemCSE: `distance(left, right)` returns `D_global`, `D_coverage`, and
  `D_final`.

### text-embedding-3-large and Gemini embedding-2

Install dependencies and provide API settings through environment variables.

PowerShell example:

```powershell
cd quantitative-linguistic-analysis\3a_similarity_distributions
python -m pip install -r requirements-embedding-cache.txt

$env:OPENAI_API_KEY = "YOUR_KEY"
$env:OPENAI_BASE_URL = "YOUR_OPTIONAL_OPENAI_COMPATIBLE_BASE_URL"
$env:GEMINI_API_KEY = $env:OPENAI_API_KEY
$env:GEMINI_API_BASE_URL = "YOUR_GEMINI_EMBEDDING_API_BASE_URL"

python precompute_text_embedding_3_large_reference_embeddings.py
python precompute_gemini_embedding_2_reference_embeddings.py
```

For a small connectivity test without scanning the whole table:

```powershell
python precompute_text_embedding_3_large_reference_embeddings.py `
  --models llama-4-scout --strategies "Strategy 0" --limit-records 2
```

Use `--cache-dir <path>` to place a test cache in an isolated directory. Use
`--dry-run` to calculate cache keys and missing-entry counts without API calls.

### SemCSE exact reproducibility

The historical SemCSE CSVs were generated in:

- Python 3.10.20
- PyTorch 2.13.0+cpu
- Transformers 5.14.1
- tokenizers 0.22.2
- NumPy 2.2.6
- safetensors 0.8.0
- model `CLAUSE-Bielefeld/SemCSE`
- verified local revision
  `d9b1d2858aafc41bc3061854332a490a41ada83e`
- CLS pooling, no L2 normalization, Euclidean distance

These versions reproduced historical document and chunk vectors bit-for-bit.
The SemCSE script pins the model revision and checks the Torch/Transformers
versions before it creates or reads a reproducible cache. The runtime versions
and model revision are also part of each new cache key, preventing accidental
mixing with vectors produced by another environment.
Use the exact environment file:

```powershell
conda create -n qla-semcse python=3.10.20 -y
conda activate qla-semcse
python -m pip install -r requirements-semcse-reproducible.txt
python precompute_semcse_reference_embeddings.py
```

On the original workstation, the already verified environment can be used
directly:

```powershell
D:\anaconda\envs\chem\python.exe precompute_semcse_reference_embeddings.py
```

Using different Torch/Transformers versions can materially change raw SemCSE
vectors even when the model name, source text, cache key, pooling, and vector
dimension are unchanged.

## Automatic local-cache fallback

Every plotting entry point follows this policy:

1. If its bundled CSV exists, read it and calculate/plot normally.
2. If a Figure 3a CSV is missing, rebuild it only from the compressed source
   table and `3a_similarity_distributions/embedding_cache/`.
3. If a Figure 3d sample-score CSV is missing, rebuild it only from the same
   source/cache plus the bundled
   `3d_mean_local_mechanistic_discrimination/input_data/descriptor_24d_values.csv`.
4. If a required local cache is absent, stop with an error naming the local
   precompute command. No parent-repository fallback is attempted.

SemCSE `D_final` is calculated from cached raw vectors as
`0.3 * D_global + 0.7 * D_coverage`. The 3a `alignment_zscore` then compares
the matched `D_final` with the other 129 reference mechanisms. The precompute
script creates vectors; the folder-local rebuild code calculates these derived
metrics.

## Directory layout

- `3a_similarity_distributions/`: row-level similarity/alignment data, cache
  scripts, compressed source data, and distribution figures.
- `3b_mean_rationale_mechanism_similarity/`: means by prediction outcome.
- `3c_transition_group_mean_trajectories/`: adjacent-strategy trajectories.
- `3d_mean_local_mechanistic_discrimination/`: candidates10 local
  discrimination, including its minimal 24-dimensional descriptor input.
