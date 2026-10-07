#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import importlib
import math
import os
import re
import socket
import sys
import time
import uuid
import unicodedata
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import parse_qs, urlparse


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = APP_DIR / "data"
RUNS_DIR = DATA_DIR / "runs"
CONFIG_PATH = DATA_DIR / "config.json"
REFERENCE_DB_PATH = DATA_DIR / "reference_db.json"
RUNS_INDEX_PATH = DATA_DIR / "runs_index.json"
BASELINE_PROMPT_PATH = DATA_DIR / "prompt_baseline.txt"
RAG_PROMPT_PATH = DATA_DIR / "prompt.txt"
# Initial default only; a temperature already saved in data/config.json takes precedence.
DEFAULT_TEMPERATURE = 0.2

DEFAULT_DESCRIPTORS = """VE3sign_D/Dt
VE1sign_Dz(v)
VE1sign_B(p)
MATS3p
GATS6i
GATS7s
P_VSA_charge_4
P_VSA_charge_7
SpMAD_EA(ri)
SpDiam_AEA(ed)
SM10_AEA(dm)
nN(CO)2
C-016
H-052
CATS2D_05_DA
CATS2D_09_DA
CATS2D_03_DL
CATS2D_05_DL
CATS2D_06_DL
CATS2D_04_AA
B09[O-O]
F05[N-N]
F07[N-O]
F10[O-O]"""

DEFAULT_PROMPT = """As an expert in nucleoside-derived gel formation, determine whether the target molecule can form a gel under the specified conditions.

Strategy: {strategy}

Retrieved reference context:
{retrieval_context}

Target molecule:
{target_json}

Return a JSON array with one object for the target molecule. Preserve the target chemical_description and canonical_smiles. For each target experiment, return solvent_additive, detailed_conditions, results ("Gel" or "No Gel"), and reasoning."""

DEFAULT_CONFIG = {
    "base_url": "https://api.openai.com/v1/",
    "api_key": "",
    "model": "gpt-4o",
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": 10000,
    "timeout": 300,
    "alvadesc_exe": "",
    "alvadesc_wrapper_path": "",
    "descriptor_names": DEFAULT_DESCRIPTORS,
    "reference_db_path": str(REFERENCE_DB_PATH),
}

STRATEGIES = {
    "S0": {
        "name": "No retrieved context",
        "include_mechanism": False,
        "condition_filter": False,
        "cross_condition": False,
    },
    "S1": {
        "name": "Max-K structural analogues",
        "include_mechanism": False,
        "condition_filter": False,
        "cross_condition": True,
    },
    "S2": {
        "name": "Condition-aware analogues",
        "include_mechanism": False,
        "condition_filter": True,
        "cross_condition": True,
    },
    "S3": {
        "name": "Condition-aware analogues + mechanism",
        "include_mechanism": True,
        "condition_filter": True,
        "cross_condition": True,
    },
}

SEPARATOR_PATTERN = re.compile(r"\s*(?:;|\+|\bor\b|\band\b|,)\s*", re.IGNORECASE)
def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    if not CONFIG_PATH.exists():
        save_json(CONFIG_PATH, DEFAULT_CONFIG)
    if not RUNS_INDEX_PATH.exists():
        save_json(RUNS_INDEX_PATH, [])


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_group_dir(temperature: float, max_k: int) -> Path:
    temperature_name = format(float(temperature), "g")
    return RUNS_DIR / f"temp={temperature_name}" / f"Max-{max_k}"


def find_run_path(run_id: str) -> Path:
    safe_id = Path(run_id).name
    legacy_path = RUNS_DIR / f"{safe_id}.json"
    if legacy_path.exists():
        return legacy_path
    matches = sorted(RUNS_DIR.glob(f"temp=*/Max-*/{safe_id}.json"))
    return matches[0] if matches else legacy_path


def load_config() -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config.update(read_json(CONFIG_PATH, {}) or {})
    # The reference database is bundled with the site so moving its directory remains portable.
    config["reference_db_path"] = str(REFERENCE_DB_PATH)
    config["descriptor_names"] = DEFAULT_DESCRIPTORS
    config.pop("prompt_template", None)
    config.pop("descriptor_library_path", None)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    clean = dict(config)
    if clean.get("api_key"):
        clean["api_key"] = "********"
    clean["reference_db_exists"] = Path(clean.get("reference_db_path") or "").exists()
    clean["reference_count"] = len(read_json(Path(clean["reference_db_path"]), []) or []) if clean["reference_db_exists"] else 0
    return clean


def descriptor_names(config: dict[str, Any]) -> list[str]:
    return [line.strip() for line in str(config.get("descriptor_names", "")).splitlines() if line.strip()]


def normalize_smiles(value: Any) -> str:
    return str(value or "").strip()


def canonicalize_smiles(value: Any) -> str:
    raw_smiles = normalize_smiles(value)
    if not raw_smiles:
        return ""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required to canonicalize the input SMILES.") from exc
    molecule = Chem.MolFromSmiles(raw_smiles)
    if molecule is None:
        raise RuntimeError(f"Invalid SMILES: {raw_smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def find_stored_descriptor_by_smiles(
    db: list[dict[str, Any]], smiles: str, names: list[str]
) -> list[float] | None:
    target_smiles = normalize_smiles(smiles)
    if not target_smiles:
        return None
    for entry in db:
        if normalize_smiles(entry.get("canonical_smiles")) == target_smiles:
            return get_stored_descriptor(entry, names)
    return None


def normalize_base_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def tokenize_condition(value: Any) -> set[str]:
    text = str(value or "").lower()
    text = re.sub(r"[()\[\]{}]", " ", text)
    parts = [part.strip() for part in SEPARATOR_PATTERN.split(text) if part.strip()]
    tokens = set(parts)
    tokens.update(re.findall(r"[a-z0-9_+\-]+", text))
    return {token for token in tokens if token}


def target_condition_tokens(target: dict[str, Any]) -> list[tuple[str, set[str]]]:
    conditions = []
    for exp in target.get("experiments", []):
        label = str(exp.get("solvent_additive") or "").strip()
        if label:
            conditions.append((label, tokenize_condition(label)))
    return conditions


def parse_experiments(text: str) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []
    jsonish = strip_json_fence(text)
    parsed = parse_jsonish_experiments(jsonish)
    if parsed is not None:
        if len(parsed) != 1:
            raise RuntimeError("Enter exactly one experiment condition for each prediction.")
        return parsed

    if jsonish[:1] in {"{", "["}:
        raise RuntimeError(
            "Target conditions / experiments looks like JSON but could not be parsed. "
            "Use a JSON array, a single JSON object, or one line per condition with '|'."
        )

    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if not parts or not parts[0]:
            continue
        rows.append(
            {
                "solvent_additive": parts[0],
                "detailed_conditions": parts[1] if len(parts) > 1 else parts[0],
            }
        )
    if len(rows) != 1:
        raise RuntimeError("Enter exactly one experiment condition for each prediction.")
    return rows


def strip_json_fence(text: str) -> str:
    stripped = text.strip().lstrip("\ufeff")
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def remove_trailing_json_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def normalize_experiment_json(data: Any) -> list[dict[str, Any]] | None:
    if isinstance(data, dict):
        experiments = data.get("experiments")
        if isinstance(experiments, list):
            return [item for item in experiments if isinstance(item, dict)]
        return [data]
    if isinstance(data, list):
        rows: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            experiments = item.get("experiments")
            if isinstance(experiments, list) and not item.get("solvent_additive"):
                rows.extend(exp for exp in experiments if isinstance(exp, dict))
            else:
                rows.append(item)
        return rows
    return None


def parse_jsonish_experiments(text: str) -> list[dict[str, Any]] | None:
    candidates = [text, remove_trailing_json_commas(text)]
    if text.startswith("{"):
        body = text.rstrip().rstrip(",")
        candidates.extend([f"[{body}]", remove_trailing_json_commas(f"[{body}]")])

    for candidate in candidates:
        try:
            normalized = normalize_experiment_json(json.loads(candidate))
            if normalized is not None:
                return normalized
        except json.JSONDecodeError:
            continue

    decoded_items = []
    decoder = json.JSONDecoder()
    idx = 0
    cleaned = remove_trailing_json_commas(text)
    while idx < len(cleaned):
        while idx < len(cleaned) and cleaned[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(cleaned):
            break
        try:
            item, end = decoder.raw_decode(cleaned, idx)
        except json.JSONDecodeError:
            return None
        decoded_items.append(item)
        idx = end
    if decoded_items:
        return normalize_experiment_json(decoded_items)
    return None


def get_stored_descriptor(entry: dict[str, Any], names: list[str]) -> list[float] | None:
    for key in ("descriptors", "descriptor_values", "descriptor_vector"):
        value = entry.get(key)
        if isinstance(value, dict):
            try:
                return [float(value[name]) for name in names]
            except Exception:
                continue
        if isinstance(value, list):
            try:
                return [float(item) for item in value]
            except Exception:
                continue
    return None


def compute_descriptors_with_alvadesc(
    smiles_list: list[str], names: list[str], alvadesc_exe: str
) -> list[list[float]]:
    if not alvadesc_exe:
        raise RuntimeError("AlvaDesc local path is not configured.")
    exe = Path(alvadesc_exe)
    if not exe.exists():
        raise RuntimeError(f"AlvaDesc executable not found: {exe}")

    locate_alvadesc_wrapper()
    from alvadesccliwrapper.alvadesc import AlvaDesc

    alva = AlvaDesc(str(exe))
    alva.set_input_SMILES(smiles_list)
    if not alva.calculate_descriptors(names):
        raise RuntimeError(f"AlvaDesc calculation failed: {alva.get_error()}")

    output_names = list(alva.get_output_descriptors())
    output_rows = alva.get_output()
    vectors = []
    for row in output_rows:
        extra = len(row) - len(output_names)
        values_by_name = {name: row[idx + max(extra, 0)] for idx, name in enumerate(output_names)}
        vectors.append([float(values_by_name[name]) for name in names])
    return vectors


def locate_alvadesc_wrapper() -> None:
    configured_path = str(load_config().get("alvadesc_wrapper_path") or "").strip()
    configured_error = ""
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if not candidate.exists():
            configured_error = f" Configured path does not exist: {candidate}."
        else:
            import_root = candidate.parent if candidate.name.lower() == "alvadesccliwrapper" else candidate
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
            try:
                importlib.import_module("alvadesccliwrapper.alvadesc")
                return
            except ModuleNotFoundError:
                configured_error = (
                    f" The alvadesccliwrapper package was not found under the configured path: {candidate}."
                )

    try:
        importlib.import_module("alvadesccliwrapper.alvadesc")
        return
    except ModuleNotFoundError:
        pass

    raise RuntimeError(
        "Cannot import alvadesccliwrapper. Configure alvadesc_wrapper_path in the web Config page, "
        "install the package in the active Python environment, or add its parent directory to PYTHONPATH."
        f"{configured_error}"
    )


def build_descriptor_matrix(
    db: list[dict[str, Any]],
    target: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[list[float]], list[float]]:
    names = descriptor_names(config)
    if not names:
        raise RuntimeError("No descriptor names configured.")

    db_vectors = [get_stored_descriptor(entry, names) for entry in db]
    target_vector = find_stored_descriptor_by_smiles(db, target.get("canonical_smiles", ""), names)
    missing_db = [idx for idx, vector in enumerate(db_vectors) if vector is None]
    if missing_db or target_vector is None:
        smiles = [db[idx]["canonical_smiles"] for idx in missing_db]
        if target_vector is None:
            smiles.append(target["canonical_smiles"])
        try:
            computed = compute_descriptors_with_alvadesc(smiles, names, config.get("alvadesc_exe", ""))
        except RuntimeError as exc:
            if target_vector is None:
                raise RuntimeError(
                    "This SMILES is not in the built-in molecule library. Configure AlvaDesc executable and "
                    f"alvadesccliwrapper paths in Config to calculate its 24D descriptors. Details: {exc}"
                ) from exc
            raise RuntimeError(
                "The active reference DB contains molecules without built-in 24D descriptors. Configure "
                f"AlvaDesc executable and alvadesccliwrapper paths in Config. Details: {exc}"
            ) from exc
        for idx, vector in zip(missing_db, computed[: len(missing_db)]):
            db_vectors[idx] = vector
        if target_vector is None:
            target_vector = computed[-1]

    return [vector for vector in db_vectors if vector is not None], target_vector


def standardize_distances(db_vectors: list[list[float]], target_vector: list[float]) -> list[float]:
    cols = len(target_vector)
    all_rows = db_vectors + [target_vector]
    means = [sum(row[col] for row in all_rows) / len(all_rows) for col in range(cols)]
    stds = []
    for col in range(cols):
        var = sum((row[col] - means[col]) ** 2 for row in all_rows) / len(all_rows)
        stds.append(math.sqrt(var) or 1.0)

    target_scaled = [(target_vector[col] - means[col]) / stds[col] for col in range(cols)]
    distances = []
    for row in db_vectors:
        scaled = [(row[col] - means[col]) / stds[col] for col in range(cols)]
        distances.append(math.sqrt(sum((scaled[col] - target_scaled[col]) ** 2 for col in range(cols))))
    return distances


def matching_experiments(entry: dict[str, Any], target_tokens: list[tuple[str, set[str]]]) -> list[dict[str, Any]]:
    if not target_tokens:
        return list(entry.get("experiments", []))
    matches = []
    for exp in entry.get("experiments", []):
        tokens = tokenize_condition(exp.get("solvent_additive", ""))
        if any(tokens & target_token for _, target_token in target_tokens):
            matches.append(exp)
    return matches


def select_references(
    db: list[dict[str, Any]],
    distances: list[float],
    target: dict[str, Any],
    strategy: str,
    max_k: int,
) -> list[dict[str, Any]]:
    if strategy == "S0":
        return []

    info = STRATEGIES[strategy]
    target_tokens = target_condition_tokens(target)
    ranked = sorted(range(len(db)), key=lambda idx: (distances[idx], int(db[idx].get("molecule_index", idx + 1))))
    target_smiles = normalize_smiles(target.get("canonical_smiles"))
    if target_smiles:
        ranked = [
            idx
            for idx in ranked
            if normalize_smiles(db[idx].get("canonical_smiles")) != target_smiles
        ]

    selected = []
    for idx in ranked:
        entry = db[idx]
        relevant = matching_experiments(entry, target_tokens) if info["condition_filter"] else list(entry.get("experiments", []))
        if info["condition_filter"] and not relevant:
            continue
        selected.append(
            {
                "entry": entry,
                "distance": distances[idx],
                "experiments": list(entry.get("experiments", [])) if info["cross_condition"] else relevant,
            }
        )
        if len(selected) >= max_k:
            break

    if len(selected) < max_k and info["condition_filter"]:
        used = {item["entry"].get("molecule_index") for item in selected}
        for idx in ranked:
            entry = db[idx]
            if entry.get("molecule_index") in used:
                continue
            selected.append({"entry": entry, "distance": distances[idx], "experiments": list(entry.get("experiments", []))})
            if len(selected) >= max_k:
                break

    return selected


def public_reference_record(ref: dict[str, Any], include_mechanism: bool) -> dict[str, Any]:
    entry = ref["entry"]
    experiments = []
    for exp in ref["experiments"]:
        item = {
            "solvent_additive": exp.get("solvent_additive", ""),
            "results": exp.get("results", ""),
            "detailed_conditions": exp.get("detailed_conditions", ""),
            "detailed_results": exp.get("detailed_results", ""),
        }
        if include_mechanism:
            item["mechanistic_explanation"] = exp.get(
                "mechanistic_explanation",
                exp.get("mechanism_explanation", ""),
            )
        experiments.append(item)
    return {
        "molecule_index": entry.get("molecule_index", ""),
        "chemical_description": entry.get("chemical_description", ""),
        "canonical_smiles": entry.get("canonical_smiles", ""),
        "chemical_distance": round(float(ref["distance"]), 6),
        "experiments": experiments,
    }


def build_retrieval_context(refs: list[dict[str, Any]], strategy: str) -> str:
    if not refs:
        return "No retrieved context."
    include_mechanism = STRATEGIES[strategy]["include_mechanism"]
    chunks = []
    for rank, ref in enumerate(refs, 1):
        record = public_reference_record(ref, include_mechanism)
        chunks.append(f"[Reference {rank} | Chemical Distance={record['chemical_distance']:.6f}]\n{json.dumps(record, ensure_ascii=False, indent=2)}")
    return "\n\n".join(chunks)


def fill_prompt(template: str, strategy: str, retrieval_context: str, target: dict[str, Any]) -> str:
    target_json = json.dumps([target], ensure_ascii=False, indent=2)
    replacements = {
        "{strategy}": f"{strategy} - {STRATEGIES[strategy]['name']}",
        "{retrieval_context}": retrieval_context,
        "{records}": retrieval_context,
        "{target_json}": target_json,
    }
    prompt = template or DEFAULT_PROMPT
    if not any(token in prompt for token in replacements):
        prompt += "\n\nRetrieved reference context:\n{retrieval_context}\n\nTarget molecule:\n{target_json}\n"
    for token, value in replacements.items():
        prompt = prompt.replace(token, value)
    return prompt


def load_run_prompt(max_k: int) -> str:
    prompt_path = BASELINE_PROMPT_PATH if max_k == 0 else RAG_PROMPT_PATH
    if not prompt_path.exists():
        raise RuntimeError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8-sig")


def call_llm(config: dict[str, Any], model: str, prompt: str) -> str:
    api_key = str(config.get("api_key", "")).strip()
    if not api_key:
        raise RuntimeError("API key is missing. Open Config, enter a valid API key, and click Save Config.")
    base_url = str(config.get("base_url", "")).strip()
    if not base_url:
        raise RuntimeError("LLM API URL is missing. Open Config, enter an OpenAI-compatible URL, and click Save Config.")
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError("LLM API URL is invalid. It must start with http:// or https://.")
    selected_model = str(model or config.get("model") or "").strip()
    if not selected_model:
        raise RuntimeError("Model name is missing. Open Config, enter a model name, and click Save Config.")

    payload = {
        "model": selected_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(config.get("temperature", DEFAULT_TEMPERATURE)),
        "max_tokens": int(config.get("max_tokens", 10000)),
    }
    req = request.Request(
        normalize_base_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=int(config.get("timeout", 300))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        provider_message = detail
        try:
            parsed_detail = json.loads(detail)
            provider_message = str(
                parsed_detail.get("error", {}).get("message")
                or parsed_detail.get("message")
                or detail
            )
        except (json.JSONDecodeError, AttributeError):
            pass
        provider_message = provider_message.strip()[:500]
        reasons = {
            400: "The LLM provider rejected the request. Check the model name, URL, and generation parameters.",
            401: "API key authentication failed. The API key is missing, invalid, expired, or belongs to another provider.",
            403: "The API key does not have permission to use this endpoint or model.",
            404: "The LLM endpoint or model was not found. Check the API URL and model name.",
            408: "The LLM provider timed out while processing the request.",
            429: "The LLM provider rate limit or account quota was exceeded. Wait and retry, or check billing/quota.",
        }
        reason = reasons.get(exc.code)
        if reason is None and 500 <= exc.code < 600:
            reason = "The LLM provider has a server-side error. Retry later or use another endpoint."
        if reason is None:
            reason = "The LLM request failed."
        suffix = f" Provider message: {provider_message}" if provider_message else ""
        raise RuntimeError(f"{reason} (HTTP {exc.code}).{suffix}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise RuntimeError(
            "The LLM request timed out. Increase Timeout seconds in Config or check the provider status."
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Cannot connect to the LLM API. Check the URL, internet connection, DNS, proxy, and TLS certificate. Details: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("The LLM provider returned a response that is not valid JSON.") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "The LLM provider response is missing choices[0].message.content. Confirm that the URL is OpenAI-compatible."
        ) from exc


def extract_json(text: str) -> Any:
    stripped = text.strip()
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def normalize_prediction_entries(extracted_json: Any) -> list[dict[str, Any]]:
    if extracted_json is None:
        return []
    if isinstance(extracted_json, dict):
        items = [extracted_json]
    elif isinstance(extracted_json, list):
        items = [item for item in extracted_json if isinstance(item, dict)]
    else:
        return []

    entries: list[dict[str, Any]] = []
    for item in items:
        common = {
            "chemical_description": item.get("chemical_description", ""),
            "canonical_smiles": item.get("canonical_smiles", ""),
        }
        experiments = item.get("experiments")
        if isinstance(experiments, list):
            for exp in experiments:
                if isinstance(exp, dict):
                    entries.append({**common, **exp})
        else:
            entries.append(item)
    return entries


def normalize_result(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"gel", "true", "yes"}:
        return "Gel"
    if text in {"no gel", "nogel", "false", "no"}:
        return "No Gel"
    return str(value or "").strip()


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "01234567890123456789+-"))
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def condition_key(value: Any) -> str:
    text = normalized_text(value)
    return re.sub(r"[^a-z0-9+\-]+", "", text)


def find_db_target(db: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any] | None:
    target_smiles = normalized_text(target.get("canonical_smiles"))
    target_name = normalized_text(target.get("chemical_description"))
    for entry in db:
        if target_smiles and normalized_text(entry.get("canonical_smiles")) == target_smiles:
            return entry
    for entry in db:
        if target_name and normalized_text(entry.get("chemical_description")) == target_name:
            return entry
    return None


def experiment_matches(query: dict[str, Any], candidate: dict[str, Any]) -> bool:
    query_additive = normalized_text(query.get("solvent_additive"))
    candidate_additive = normalized_text(candidate.get("solvent_additive"))
    if query_additive and candidate_additive and query_additive == candidate_additive:
        return True
    if query_additive and candidate_additive and condition_key(query_additive) == condition_key(candidate_additive):
        return True

    query_conditions = normalized_text(query.get("detailed_conditions"))
    candidate_conditions = normalized_text(candidate.get("detailed_conditions"))
    if query_conditions and candidate_conditions:
        return query_conditions == candidate_conditions or query_conditions in candidate_conditions or candidate_conditions in query_conditions

    query_tokens = tokenize_condition(query.get("solvent_additive", ""))
    candidate_tokens = tokenize_condition(candidate.get("solvent_additive", ""))
    return bool(query_tokens and candidate_tokens and query_tokens & candidate_tokens)


def build_expected_from_db(target: dict[str, Any], db_entry: dict[str, Any]) -> list[dict[str, Any]]:
    db_experiments = [exp for exp in db_entry.get("experiments", []) if str(exp.get("results", "")).strip()]
    target_experiments = target.get("experiments", [])
    if not target_experiments:
        source_experiments = db_experiments
    else:
        source_experiments = []
        used: set[int] = set()
        for target_exp in target_experiments:
            for idx, db_exp in enumerate(db_experiments):
                if idx in used:
                    continue
                if experiment_matches(target_exp, db_exp):
                    source_experiments.append(db_exp)
                    used.add(idx)
                    break
        if not source_experiments:
            source_experiments = db_experiments

    return [
        {
            "index": idx,
            "solvent_additive": str(exp.get("solvent_additive", "")).strip(),
            "detailed_conditions": str(exp.get("detailed_conditions", "")).strip(),
            "expected_result": normalize_result(exp.get("results")),
            "expected_source": "DB.json",
        }
        for idx, exp in enumerate(source_experiments)
    ]


def compute_accuracy(target: dict[str, Any], extracted_json: Any, db: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    db_entry = find_db_target(db or [], target)
    expected = build_expected_from_db(target, db_entry) if db_entry else []
    expected_source = expected[0]["expected_source"] if expected else None
    if not expected:
        return {
            "available": False,
            "correct": None,
            "total": 0,
            "accuracy": None,
            "expected_source": None,
            "message": "No expected results were found for this target in the built-in DB.",
            "details": [],
        }

    predictions = normalize_prediction_entries(extracted_json)
    used_prediction_indices: set[int] = set()
    details = []
    correct = 0

    for item in expected:
        match_idx = None
        for idx, pred in enumerate(predictions):
            if idx in used_prediction_indices:
                continue
            if str(pred.get("solvent_additive", "")).strip() == item["solvent_additive"]:
                match_idx = idx
                break
        if match_idx is None and item["index"] < len(predictions):
            match_idx = item["index"]

        predicted_result = ""
        if match_idx is not None:
            used_prediction_indices.add(match_idx)
            predicted_result = normalize_result(predictions[match_idx].get("results"))

        is_correct = predicted_result == item["expected_result"]
        correct += int(is_correct)
        details.append(
            {
                "solvent_additive": item["solvent_additive"],
                "detailed_conditions": item["detailed_conditions"],
                "expected_result": item["expected_result"],
                "predicted_result": predicted_result or "Missing",
                "correct": is_correct,
                "expected_source": item["expected_source"],
            }
        )

    total = len(expected)
    return {
        "available": True,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "expected_source": expected_source,
        "message": f"{correct}/{total} correct using {expected_source}",
        "details": details,
    }


def run_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    strategy = payload.get("strategy", "S2")
    if strategy not in STRATEGIES:
        raise RuntimeError(f"Unknown strategy: {strategy}")

    target = {
        "chemical_description": payload.get("chemical_description", "").strip(),
        "canonical_smiles": canonicalize_smiles(payload.get("canonical_smiles", "")),
        "experiments": parse_experiments(payload.get("target_experiments", "")),
    }
    if not target["chemical_description"] or not target["canonical_smiles"]:
        raise RuntimeError("Target molecule name and SMILES are required.")

    db_path = Path(config.get("reference_db_path") or REFERENCE_DB_PATH)
    db = read_json(db_path, [])
    if not isinstance(db, list):
        db = []
    if strategy != "S0":
        if not db:
            raise RuntimeError("Reference DB is empty. Import a reference DB in Config first.")

    max_k = int(payload.get("max_k", 6))
    if max_k < 0:
        raise RuntimeError("Max-k must be 0 or greater.")
    model = payload.get("model") or config.get("model")

    refs = []
    distances = []
    if strategy != "S0" and max_k > 0:
        db_vectors, target_vector = build_descriptor_matrix(db, target, config)
        distances = standardize_distances(db_vectors, target_vector)
        refs = select_references(db, distances, target, strategy, max_k)

    retrieval_context = build_retrieval_context(refs, strategy)
    prompt = fill_prompt(load_run_prompt(max_k), strategy, retrieval_context, target)

    raw_output = call_llm(config, model, prompt)
    parsed_json = extract_json(raw_output)
    accuracy = compute_accuracy(target, parsed_json, db)

    run_id = next_run_id(target["chemical_description"])
    record = {
        "id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": strategy,
        "strategy_name": STRATEGIES[strategy]["name"],
        "model": model,
        "max_k": max_k,
        "temperature": float(config.get("temperature", DEFAULT_TEMPERATURE)),
        "target": target,
        "selected_references": [
            public_reference_record(ref, STRATEGIES[strategy]["include_mechanism"]) for ref in refs
        ],
        "retrieval_context": retrieval_context,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_json": parsed_json,
        "accuracy": accuracy,
        "config_snapshot": {**public_config(config), "api_key": "********"},
    }
    save_run(record)
    return record


def save_run(record: dict[str, Any]) -> None:
    run_path = run_group_dir(record["temperature"], record["max_k"]) / f"{record['id']}.json"
    save_json(run_path, record)
    index = read_json(RUNS_INDEX_PATH, []) or []
    if isinstance(index, dict):
        index = [index]
    index.insert(
        0,
        {
            "id": record["id"],
            "created_at": record["created_at"],
            "strategy": record["strategy"],
            "model": record["model"],
            "max_k": record["max_k"],
            "temperature": record["temperature"],
            "path": run_path.relative_to(RUNS_DIR).as_posix(),
            "target": record["target"]["chemical_description"],
            "accuracy": record.get("accuracy", {}),
        },
    )
    save_json(RUNS_INDEX_PATH, index[:300])


def sanitize_id_part(text: str, max_len: int = 42) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return (value or "target")[:max_len].strip("._-") or "target"


def next_run_id(target_name: str) -> str:
    prefix = sanitize_id_part(target_name)
    existing_numbers = []
    for path in RUNS_DIR.rglob(f"{prefix}_R*.json"):
        match = re.search(r"_R(\d+)\.json$", path.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    return f"{prefix}_R{(max(existing_numbers) if existing_numbers else 0) + 1}"


def delete_run(run_id: str) -> dict[str, Any]:
    safe_id = Path(run_id).name
    run_path = find_run_path(safe_id)
    if run_path.exists():
        run_path.unlink()
    index = read_json(RUNS_INDEX_PATH, []) or []
    if isinstance(index, dict):
        index = [index]
    index = [item for item in index if item.get("id") != safe_id]
    save_json(RUNS_INDEX_PATH, index)
    return {"deleted": safe_id}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/config":
                self.send_json(public_config(load_config()))
            elif parsed.path == "/api/runs":
                self.send_json(read_json(RUNS_INDEX_PATH, []) or [])
            elif parsed.path.startswith("/api/runs/"):
                run_id = parsed.path.rsplit("/", 1)[-1]
                self.send_json(read_json(find_run_path(run_id), {}) or {})
            elif parsed.path == "/api/status":
                config = load_config()
                self.send_json(public_config(config))
            else:
                self.send_static(parsed.path)
        except Exception as exc:
            self.send_error_json(str(exc))

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            payload = self.read_body_json()
            if parsed.path == "/api/config":
                config = load_config()
                api_key_value = payload.get("api_key")
                config.update({key: value for key, value in payload.items() if key in DEFAULT_CONFIG})
                if api_key_value == "********":
                    config["api_key"] = load_config().get("api_key", "")
                save_json(CONFIG_PATH, config)
                self.send_json(public_config(config))
            elif parsed.path == "/api/run":
                self.send_json(run_strategy(payload))
            else:
                self.send_error_json(f"Unknown endpoint: {parsed.path}", status=404)
        except Exception as exc:
            self.send_error_json(str(exc))

    def do_DELETE(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/runs/"):
                run_id = parsed.path.rsplit("/", 1)[-1]
                self.send_json(delete_run(run_id))
            else:
                self.send_error_json(f"Unknown endpoint: {parsed.path}", status=404)
        except Exception as exc:
            self.send_error_json(str(exc))

    def read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def send_static(self, path: str) -> None:
        if path in ("", "/"):
            file_path = STATIC_DIR / "index.html"
        else:
            file_path = (STATIC_DIR / path.lstrip("/")).resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                self.send_error_json("Forbidden", status=403)
                return
        if not file_path.exists():
            self.send_error_json("Not found", status=404)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(file_path.suffix, "application/octet-stream")
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status=status)


def main() -> None:
    ensure_dirs()
    port = int(os.environ.get("RAG_STRATEGY_SITE_PORT", "8777"))
    host = os.environ.get("RAG_STRATEGY_SITE_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"RAG Strategy Site running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
