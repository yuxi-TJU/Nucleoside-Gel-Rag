# Nucleoside Gel RAG Evaluation Dashboard

A self-contained static dashboard for exploring and comparing evaluation
results from retrieval-augmented generation (RAG) strategies for nucleoside gel
prediction.

The packaged dashboard runs entirely in the browser. It does not require a
Python server, an API key, or a database.

## Features

- Filter results by model, RAG strategy, Max-k, round, and molecule.
- Compare accuracy across models and retrieval strategies.
- Inspect 12,900 molecule-round evaluation records.
- Compare parsed model outputs for corresponding experimental conditions.
- Open a complete offline detail page for every record, including the original
  model input and output text for each condition.

## Open Locally

Download the complete `nucleoside-gel-rag-evaluation-dashboard` folder and
open `index.html` in a modern browser.

The evaluation dataset is embedded in `index.html`, so the dashboard also
works through a local `file://` URL. No installation command is required.

The public package is approximately 443 MB and contains 12,900 detail pages.
Allow the download or clone to finish before opening the dashboard.

## GitHub Pages

When this folder is stored in a GitHub repository, it can be published with
GitHub Pages:

1. Open the repository's **Settings > Pages**.
2. Select the branch and root folder used for the site.
3. Open the deployed path ending in
   `/nucleoside-gel-rag-evaluation-dashboard/`.

GitHub Pages configuration is optional. The downloaded dashboard provides the
same functionality when `index.html` is opened locally.

## Evaluation Snapshot

The packaged snapshot contains:

- 50 model/strategy/Max-k evaluation configurations;
- 12,900 molecule-round records;
- 12,900 corresponding offline detail pages;
- Rounds 1-3 and RAG strategies S0-S3;
- molecule-level accuracy, retrieval metadata, conditions, parsed outputs,
  prompts, and complete model input/output text where applicable.

All detail paths and condition panels were validated against the generated
dataset before packaging.

## Included Files

```text
nucleoside-gel-rag-evaluation-dashboard/
|-- index.html                  Dashboard page with embedded data
|-- app.js                      Filtering, comparison, and detail behavior
|-- styles.css                  Dashboard styles
|-- details/                    12,900 molecule-level detail pages
|-- .gitignore
`-- README.md
```

The complete evaluation dataset is embedded directly in `index.html`. The
browser does not fetch a separate JSON file at runtime.

This repository is a pre-generated static distribution. It intentionally does
not include the private source experiment directories or maintainer-side build
scripts. Users only need the files listed above to browse the dashboard.

## Data and Privacy

The dashboard contains generated model inputs, outputs, parsed JSON, aggregate
metrics, molecule descriptions, and experimental conditions. It contains no
API key and makes no request to an LLM provider. Review the evaluation content
before publishing if any source experiments are private.
