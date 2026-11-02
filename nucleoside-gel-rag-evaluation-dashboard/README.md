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
