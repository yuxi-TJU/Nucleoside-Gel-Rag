# Nucleoside Gel Prediction

This package is a local web application for predicting whether a nucleoside-derived molecule forms a gel under one experimental condition. It uses the bundled reference database, fixed 24D molecular descriptors, and RAG strategies S0-S3.

## Quick Start

Requirements:

- Python 3.10 or later
- RDKit

Start the local web server from this directory:

```powershell
python server.py
```

Open `http://127.0.0.1:8777` in a browser. Stop the server with `Ctrl+C`.

Keep the terminal running while using the site. If the browser reports that the
connection was refused, the Python server is not running (or port 8777 is being
used by another process). Start `python server.py` first, wait for the
`RAG Strategy Site running` message, and then reload the page.

Do not open `static/index.html` directly. The HTML is only the browser client;
predictions, configuration, RDKit canonicalization, retrieval, LLM requests,
and run history are provided by `server.py` through `/api/*`. A `file://` page
has no local API server to answer those requests.

Install RDKit with Conda if needed:

```powershell
conda install -c conda-forge rdkit
```

No additional third-party Python package is required by the web server beyond RDKit.

## API Configuration

Open `Config` in the web interface, enter your OpenAI-compatible API URL, API
key, model name, and generation settings, then click `Save Config`.

On the first launch, the server automatically creates `data/config.json` with
an empty API key. You do not need to create or download this file. After you
save the form, the file stores settings only on your computer. It is excluded
from Git by `.gitignore` because it may contain credentials. Do not commit or
share your generated `config.json`.

## Running a Prediction

Each run accepts exactly one molecule and one experimental condition.

1. Enter the chemical description.
2. Enter a SMILES string. The application canonicalizes it with RDKit before matching and retrieval.
3. Enter one condition as either a JSON object or one line in the form `additive | detailed conditions`.
4. Select S0-S3 and set `Max-k`.
5. Click `Run Strategy`.

The application displays the model output, parsed JSON, prompt, retrieved references, and available accuracy evaluation. Run records are stored in `data/runs/` and can be viewed from `History`.

## Strategies

| Strategy | Retrieval | Condition-aware | Mechanism explanation | Reference conditions |
| --- | --- | --- | --- | --- |
| S0 | None | No | Not applicable | Not applicable |
| S1 | Nearest molecules by 24D distance | No | Hidden | All conditions |
| S2 | Nearest molecules by 24D distance | Yes | Hidden | All conditions |
| S3 | Nearest molecules by 24D distance | Yes | Included | All conditions |

S2 and S3 first prefer reference molecules sharing condition tokens with the target. If fewer than `Max-k` matches are available, the remaining slots are filled by the nearest molecules. The target molecule itself is excluded from retrieval.

## Prompt Selection

The original prompts are bundled in `data/prompt_baseline.txt` and `data/prompt.txt`.

- `Max-k = 0`: uses `prompt_baseline.txt` and sends no retrieved references.
- `Max-k > 0`: uses `prompt.txt` and sends references according to the selected strategy.

The prompt text is kept unchanged from the source prompt files.

## Descriptors and AlvaDesc Fallback

`data/reference_db.json` was updated from the latest root-level `DB.json` experimental records while preserving the existing 24D descriptor values for the 86 built-in molecules.

The 24D descriptor set is fixed by the application and is not editable in the web interface. Built-in molecules use their stored descriptor values. For an unknown molecule, configure both the AlvaDesc executable path and the `alvadesccliwrapper` path in `Config`; the application will calculate the same 24D descriptor set as a fallback.

The application does not scan personal folders such as OneDrive or Documents.
It loads `alvadesccliwrapper` only from the path saved in `Config`, the active
Python environment, or `PYTHONPATH`.

## Directory Layout

```text
nucleoside-gel-rag-webapp/
|-- server.py
|-- static/
|-- data/
|   |-- reference_db.json
|   |-- prompt_baseline.txt
|   `-- prompt.txt
`-- .gitignore
```

The following runtime files are created automatically and intentionally are
not included in the repository:

```text
data/config.json       Local LLM settings and API key
data/runs_index.json   Local history index
data/runs/             Local prediction records
```
