# Nucleoside Gel Prediction with Retrieval-Augmented Generation

## Overview

This repository brings together a local prediction application, precomputed evaluation results, and analysis code. It supports exploration of prediction accuracy, semantic alignment between model rationales and literature-derived mechanisms, and overlap among retrieved reference sets.

## Interactive Tools

| Tool | What you can explore |
| --- | --- |
| [Prediction evaluation dashboard](nucleoside-gel-rag-evaluation-dashboard/README.md) | Compare accuracy across models, retrieval strategies, and retrieval depths; inspect experimental conditions and model inputs and outputs. |
| [Rationale similarity dashboard](nucleoside-gel-rag-text-embedding-3-large-similarity-dashboard/README.md) | Explore `text-embedding-3-large` similarity scores, compare strategies, and inspect individual molecule–experiment samples. |
| [Gel prediction application](nucleoside-gel-rag-webapp/README.md) | Submit a molecule and experimental condition, select a retrieval strategy, and inspect predictions, retrieved examples, and model responses. |

The two dashboards display precomputed results and require no installation or API credentials. Download or clone the repository, then open `index.html` in the corresponding folder, keeping its `details/` folder alongside it.

The prediction application requires Python, RDKit, and an OpenAI-compatible language-model API. See its [setup guide](nucleoside-gel-rag-webapp/README.md) to get started.

## Analysis and Reproducibility

### Reference-Set Overlap

The [reference overlap analysis](overlap-summary/README.md) compares descriptor-based and Morgan-fingerprint retrieval, with and without condition-aware selection. It includes the data, scripts, and summary tables for Jaccard overlap and same-literature statistics at retrieval depths of **3, 5, and 6**.

### Quantitative Linguistic Analysis

The [quantitative linguistic analysis](quantitative-linguistic-analysis/README.md) provides data and scripts for four views of model rationales: similarity distributions, mean rationale–mechanism similarity by prediction outcome, trajectories between adjacent retrieval strategies, and local mechanistic discrimination.

The bundled CSV files support figure reproduction without embedding API calls. Compressed source records and embedding-generation scripts are also provided for recalculation from the underlying text. Dependencies and commands are documented in each analysis folder.
