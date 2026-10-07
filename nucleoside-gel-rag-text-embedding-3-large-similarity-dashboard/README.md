# Nucleoside-Gel RAG `text-embedding-3-large` Similarity Dashboard

This directory contains a self-contained, interactive dashboard for comparing the semantic similarity of nucleoside-gel RAG outputs across retrieval strategies and language models.

The displayed similarity values were calculated with OpenAI's `text-embedding-3-large` model. The website is a pre-generated static snapshot: viewing it does not call the OpenAI API and does not require an API key, Python environment, package installation, or web server.

## Included Results

- Five output models: `grok-4.3`, `deepseek-v3.2-think`, `gemini-3.1-flash-lite`, `gpt-4o`, and `llama-4-scout`
- Four RAG strategies: Strategy 0 through Strategy 3

## Open the Dashboard

1. Download or clone the repository.
2. Keep `index.html` and the `details/` directory together in their original relative locations.
3. Open `index.html` with a current version of Chrome, Edge, or Firefox.
4. Hover over a point to inspect its summary, and click it to open the corresponding model/sample detail page.

The dashboard can also be served by any static web server or GitHub Pages. Opening `index.html` directly is sufficient for local use.

## How to Read the Charts

- Each panel represents one language model.
- The horizontal axis shows semantic similarity to the literature-derived reference text.
- The vertical axis lists Strategy 0 through Strategy 3.
- Each violin summarizes the score distribution for one strategy.
- Each point represents one molecule-experiment sample; point color indicates prediction correctness.
- Connecting lines show how the same sample changes across strategies.
- Clicking a point opens its detailed outputs, experimental context, reference text, predictions, and similarity scores.

## Similarity Definition

For each model output and reference pair:

```text
S_final = 0.3 * S_global + 0.7 * S_coverage
```

- `S_global` is the cosine similarity between the normalized document embeddings.
- `S_coverage` is the mean of the two token-weighted, best-match chunk coverage scores, calculated in both text directions.
- `semantic_similarity` in this dashboard is the mean `S_final` value across the available rounds.
- The reference text combines the experiment's `detailed_results` and `mechanistic_explanation` fields. Samples without a mechanistic explanation are not included.

Higher values indicate closer semantic alignment with the reference text. They do not by themselves establish scientific correctness or experimental validity.

## Directory Layout

```text
nucleoside-gel-rag-text-embedding-3-large-similarity-dashboard/
|-- index.html
|-- details/
`-- README.md
```

## Sharing and Deployment

Share or deploy the complete directory. `index.html` uses relative links to files in `details/`; copying only the main page will leave all point-detail links broken.
