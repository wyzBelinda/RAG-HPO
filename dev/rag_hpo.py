#!/usr/bin/env python3
"""
RAG-HPO main pipeline: extract HPO terms from clinical notes using LLM + RAG.

Credential priority (highest to lowest):
    1. Command-line arguments (--api-key, --base-url, --model)
    2. .env file (copy .env.example to .env and fill in values)
    3. Environment variables (RAG_HPO_API_KEY, RAG_HPO_BASE_URL, ...)
    4. Existing LLaMa-4-Scout.flag file
    5. Interactive prompt

Usage:
    python dev/rag_hpo.py --csv Test_Cases.csv
"""

import os
import sys
import time
import json
import re
import subprocess
import shutil
import argparse
import unicodedata
from contextlib import contextmanager
import traceback

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

import tiktoken
import pandas as pd
import numpy as np
import faiss
from fastembed import TextEmbedding
from rapidfuzz import fuzz as rfuzz
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from http_utils import build_retry_session, request_with_retry

# =========================== Constants ===========================

FLAG_FILE = "LLaMa-4-Scout.flag"
PROMPTS_FILE = "system_prompts.json"
META_PATH = "hpo_meta.json"
VEC_PATH = "hpo_embedded.npz"

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

TEMP_FILES = {
    "input": os.path.join(TEMP_DIR, "temp_input.pkl"),
    "combined": os.path.join(TEMP_DIR, "temp_combined_results.pkl"),
    "exact": os.path.join(TEMP_DIR, "temp_exact_matches.pkl"),
    "non_exact": os.path.join(TEMP_DIR, "temp_non_exact_matches.pkl"),
    "final": os.path.join(TEMP_DIR, "temp_final_result.pkl"),
    "responses": "responses_backup.pkl",
}


# =========================== Logger ===========================

class Logger:
    def __init__(self):
        self.printed_messages = set()

    def log(self, msg, once=False):
        msg_hash = hash(msg)
        if once and msg_hash in self.printed_messages:
            return
        self.printed_messages.add(msg_hash)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


logger = Logger()


# =========================== LLM Client ===========================

class LLMClient:
    def __init__(
        self,
        api_key,
        base_url,
        model_name="llama3-groq-70b-8192-tool-use-preview",
        max_tokens_per_day=500000,
        max_queries_per_minute=30,
        temperature=0.7,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.temperature = temperature
        self.connect_timeout = 10
        self.read_timeout = 120
        self.total_tokens_used = 0
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Normalize to a chat-completions endpoint while still allowing the caller
        # to provide either a root base URL or a full endpoint URL.
        base = (base_url or "").rstrip("/")
        if base.endswith("/chat/completions"):
            self.endpoint = base
        elif base:
            self.endpoint = f"{base}/chat/completions"
        else:
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.session = build_retry_session()
        try:
            self.encoder = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def query(self, user_input, system_message, max_retries=5):
        tokens_ui = len(self.encoder.encode(user_input))
        tokens_sys = len(self.encoder.encode(system_message))
        estimated = tokens_ui + tokens_sys
        if self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Token limit exceeded for the day.")
        time.sleep((60 / max(self.max_queries_per_minute, 1)) + 0.1)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input},
            ],
            "temperature": self.temperature,
        }
        resp = request_with_retry(
            self.session,
            "POST",
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=(self.connect_timeout, self.read_timeout),
            max_retries=max_retries,
            logger=logger.log,
        )
        if not resp.ok:
            logger.log(f"[API ERROR] {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        try:
            result = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON response from LLM: {resp.text[:500]}") from e
        if "usage" in result and "total_tokens" in result["usage"]:
            self.total_tokens_used += result["usage"]["total_tokens"]
        else:
            self.total_tokens_used += estimated
        choices = result.get("choices") or []
        return choices[0].get("message", {}).get("content", "") if choices else ""


# =========================== Prompt Loading ===========================

def load_prompts(file_path=PROMPTS_FILE):
    if not os.path.exists(file_path):
        logger.log(f"Error: Prompt file '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, "r") as f:
        return json.load(f)


# =========================== Environment Setup ===========================

def _resolve_config_value(cli_val, env_key, default=None):
    """Resolve a config value using priority: CLI > env var > default."""
    if cli_val is not None:
        return cli_val
    env_val = os.getenv(env_key)
    if env_val is not None:
        return env_val
    return default


def check_and_initialize(args):
    """Initialize LLM client. Priority: CLI > .env/os.environ > flag file > interactive."""
    # 1) Try CLI args + env vars first
    api_key = _resolve_config_value(args.api_key, "RAG_HPO_API_KEY")
    base_url = _resolve_config_value(args.base_url, "RAG_HPO_BASE_URL")
    model_name = _resolve_config_value(args.model, "RAG_HPO_MODEL")
    max_tokens = _resolve_config_value(args.max_tokens, "RAG_HPO_MAX_TOKENS_PER_DAY")
    max_qpm = _resolve_config_value(args.max_qpm, "RAG_HPO_MAX_QUERIES_PER_MINUTE")
    temp = _resolve_config_value(args.temperature, "RAG_HPO_TEMPERATURE")

    if api_key:
        return LLMClient(
            api_key=api_key,
            base_url=base_url or "https://api.groq.com/openai/v1/chat/completions",
            model_name=model_name or "llama3-groq-70b-8192-tool-use-preview",
            max_tokens_per_day=int(max_tokens) if max_tokens else 500000,
            max_queries_per_minute=int(max_qpm) if max_qpm else 30,
            temperature=float(temp) if temp else 0.7,
        )

    # 2) Try legacy flag file
    if os.path.exists(FLAG_FILE):
        with open(FLAG_FILE, "r") as f:
            config = json.load(f)
        print(f"[INFO] Using config from {FLAG_FILE}")
        return LLMClient(
            api_key=config["api_key"],
            base_url=config.get("base_url", ""),
            model_name=config.get("model_name", ""),
            max_tokens_per_day=config.get("max_tokens_per_day", 500000),
            max_queries_per_minute=config.get("max_queries_per_minute", 30),
            temperature=config.get("temperature", 0.7),
        )

    # 3) Interactive fallback
    print("No credentials found in CLI args, .env, or environment variables.")
    print("Tip: copy .env.example to .env and fill in your values to skip this prompt.\n")
    api_key = input("Enter your API key (required): ").strip()
    base_url = input(
        "Enter base URL (default https://api.groq.com/openai/v1/chat/completions): "
    ) or "https://api.groq.com/openai/v1/chat/completions"
    model_name = input(
        "Enter model name (default llama3-groq-70b-8192-tool-use-preview): "
    ) or "llama3-groq-70b-8192-tool-use-preview"
    max_tokens = input("Max tokens/day (default 500000): ")
    max_tokens = int(max_tokens) if max_tokens else 500000
    max_qpm = input("Max queries/minute (default 30): ")
    max_qpm = int(max_qpm) if max_qpm else 30
    temp = input("Temperature (0.0-1.0, default 0.2): ")
    try:
        temp = float(temp) if temp else 0.2
    except ValueError:
        temp = 0.7

    client = LLMClient(api_key, base_url, model_name, max_tokens, max_qpm, temp)
    with open(FLAG_FILE, "w") as f:
        json.dump(
            {
                "api_key": api_key,
                "base_url": base_url,
                "model_name": model_name,
                "max_tokens_per_day": max_tokens,
                "max_queries_per_minute": max_qpm,
                "temperature": temp,
            },
            f,
        )
    return client


# =========================== Subprocess Helper ===========================

@contextmanager
def managed_subprocess(*args, **kwargs):
    proc = subprocess.Popen(*args, **kwargs)
    try:
        yield proc
    finally:
        proc.terminate()
        proc.wait()


# =========================== Text Cleaning ===========================

PAT = re.compile(r"\(.*?\)")


def clean_text(txt: str) -> str:
    return PAT.sub("", txt or "").replace("_", " ").lower().strip()


def clean_note(text: str) -> str:
    text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================== Embeddings & FAISS ===========================

def initialize_embeddings_model(
    use_sbert: bool = True,
    sbert_model: str = "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb",
    bge_model: str = "BAAI/bge-small-en-v1.5",
):
    try:
        if use_sbert:
            return SentenceTransformer(sbert_model)
        return TextEmbedding(model_name=bge_model)
    except Exception as e:
        logger.log(f"[FATAL] Could not initialize embedding model: {e}")
        sys.exit(1)


def load_vector_db(meta_path=META_PATH, vec_path=VEC_PATH):
    if not os.path.exists(meta_path) or not os.path.exists(vec_path):
        logger.log(f"[FATAL] DB files not found: {meta_path}, {vec_path}")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        combined = json.load(f)
        constants = combined.get("constants", {})
        entries = combined.get("entries", [])

    arr = np.load(vec_path)
    emb_matrix = arr["emb"].astype(np.float32)

    if len(entries) != emb_matrix.shape[0]:
        logger.log(
            f"[WARN] Metadata/embedding row mismatch: {len(entries)} vs {emb_matrix.shape[0]}"
        )

    docs = []
    for entry, vec in zip(entries, emb_matrix):
        hp_id = entry.get("hp_id")
        const = constants.get(hp_id, {})
        doc = {
            "hp_id": hp_id,
            "info": entry.get("info"),
            "lineage": const.get("lineage"),
            "organ_system": const.get("organ_system"),
            "direction": entry.get("direction"),
            "depth": const.get("depth"),
            "parent_count": const.get("parent_count"),
            "child_count": const.get("child_count"),
            "descendant_count": const.get("descendant_count"),
            "embedding": vec,
        }
        docs.append(doc)
    return docs, emb_matrix


def create_faiss_index(emb_matrix: np.ndarray, metric: str = "cosine"):
    dim = emb_matrix.shape[1]
    if metric == "cosine":
        faiss.normalize_L2(emb_matrix)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(emb_matrix)
    return index


def embed_query(text: str, model, metric: str = "cosine"):
    if hasattr(model, "encode"):
        vec = model.encode(text, convert_to_numpy=True)
    else:
        vec = np.array(list(model.embed([text]))[0], dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if metric == "cosine":
        faiss.normalize_L2(vec)
    return vec


# =========================== Phenotype Processing ===========================

def _collect_metadata_best(
    phrase: str,
    query_vec: np.ndarray,
    index: faiss.Index,
    docs: list,
    top_k: int = 500,
    similarity_threshold: float = 0.35,
    min_unique: int = 15,
    max_unique: int = 20,
):
    clean_tokens = set(re.findall(r"\w+", phrase.lower()))
    dists, idxs = index.search(query_vec, top_k)
    sims, indices = dists[0], idxs[0]

    seen_hp = set()
    results = []

    for sim, idx in sorted(zip(sims, indices), key=lambda x: x[0], reverse=True):
        if len(results) >= max_unique:
            break
        doc = docs[idx]
        hp = doc.get("hp_id")
        if not hp or hp in seen_hp:
            continue
        info = doc.get("info", "") or ""
        token_overlap = bool(clean_tokens & set(re.findall(r"\w+", info.lower())))
        if token_overlap or sim >= similarity_threshold or len(results) < min_unique:
            seen_hp.add(hp)
            results.append(
                {
                    "hp_id": hp,
                    "phrase": info,
                    "definition": doc.get("definition"),
                    "organ_system": doc.get("organ_system"),
                    "similarity": float(sim),
                }
            )
    return results


def split_exact_nonexact(df: pd.DataFrame, hpo_term_col="HPO_Term"):
    if "category" not in df.columns:
        raise KeyError(f"[FATAL] 'category' missing; columns: {df.columns.tolist()}")
    if hpo_term_col not in df.columns:
        raise KeyError(f"[FATAL] '{hpo_term_col}' missing; columns: {df.columns.tolist()}")
    df_ab = df[df["category"] == "Abnormal"]
    exact_df = df_ab.dropna(subset=[hpo_term_col]).copy()
    non_exact_df = df_ab[df_ab[hpo_term_col].isna()].copy()
    return exact_df, non_exact_df


def process_findings(
    findings,
    clinical_note: str,
    embeddings_model,
    index,
    docs,
    metric: str = "cosine",
    keep_top: int = 15,
):
    sentences = [s.strip() for s in clinical_note.split(".") if s.strip()]
    rows = []

    for f in findings:
        phrase = f.get("phrase", "").strip()
        category = f.get("category", "")
        if not phrase:
            continue

        qv = embed_query(phrase, embeddings_model, metric=metric)
        unique_metadata = _collect_metadata_best(
            phrase=phrase,
            query_vec=qv,
            index=index,
            docs=docs,
            top_k=500,
            similarity_threshold=0.35,
            min_unique=keep_top,
            max_unique=keep_top,
        )

        fw = set(re.findall(r"\b\w+\b", phrase.lower()))
        best_sent, best_score = None, 0
        for s in sentences:
            sw = set(re.findall(r"\b\w+\b", s.lower()))
            score = len(fw & sw)
            if score > best_score:
                best_score, best_sent = score, s

        rows.append(
            {
                "phrase": phrase,
                "category": category,
                "unique_metadata": unique_metadata,
                "original_sentence": best_sent,
                "patient_id": f.get("patient_id"),
            }
        )

    return pd.DataFrame(rows)


# =========================== LLM Response Parsing ===========================

def clean_and_parse(s: str):
    try:
        m = re.search(r"\{.*\}", s, flags=re.S)
        js_str = m.group(0) if m else s.strip()
        return json.loads(js_str)
    except Exception:
        return None


def extract_findings(response: str) -> list:
    if not response:
        return []
    parsed = clean_and_parse(response)
    if not isinstance(parsed, dict):
        return []
    return parsed.get("phenotypes", [])


# =========================== Single-Row Processing ===========================

def process_row(clinical_note, system_message, embeddings_model, index, embedded_documents):
    clinical_note = clean_note(clinical_note)
    raw = llm_client.query(clinical_note, system_message)
    findings = extract_findings(raw)

    if not findings:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                findings = parsed
        except Exception:
            pass

    if not findings:
        try:
            matches = re.findall(r"\{[^\}]*\}", raw)
            findings = []
            for m in matches:
                try:
                    d = json.loads(m)
                    if "phrase" in d and "category" in d:
                        findings.append(d)
                except Exception:
                    continue
        except Exception:
            pass

    findings = [f for f in findings if isinstance(f, dict) and f.get("category") == "Abnormal"]

    required_cols = ["phrase", "category", "unique_metadata", "original_sentence", "patient_id"]
    if not findings:
        return pd.DataFrame(columns=required_cols)

    df = process_findings(findings, clinical_note, embeddings_model, index, embedded_documents)
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


# =========================== HPO Term Extraction ===========================

def _iter_term_hp(item):
    if isinstance(item, str):
        try:
            d = json.loads(item)
        except Exception:
            return
    elif isinstance(item, dict):
        d = item
    else:
        return
    if "hp_id" in d:
        term_text = d.get("info") or d.get("label")
        hp = d["hp_id"]
        if hp and term_text:
            yield term_text, hp
        return
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("HP:"):
            yield k, v


def build_cluster_index(metadata_list):
    idx = defaultdict(lambda: defaultdict(list))
    for entry in metadata_list:
        for term, hp in _iter_term_hp(entry):
            ct = clean_text(term)
            if not ct:
                continue
            toks = ct.split()
            sig = " ".join(sorted(toks))
            idx[sig][len(toks)].append(hp)
    return idx


def extract_hpo_term(phrase, metadata_list, cluster_index):
    if not metadata_list or (isinstance(metadata_list, float) and pd.isna(metadata_list)):
        return None
    cp = clean_text(phrase)
    if not cp:
        return None
    toks = cp.split()
    sig = " ".join(sorted(toks))
    if sig in cluster_index and len(toks) in cluster_index[sig]:
        return cluster_index[sig][len(toks)][0]

    pairs = []
    for entry in metadata_list:
        for term, hp in _iter_term_hp(entry):
            ct = clean_text(term)
            if ct:
                pairs.append((ct, hp))

    pset = set(toks)
    for ct, hp in pairs:
        if set(ct.split()) == pset:
            return hp
    for ct, hp in pairs:
        if ct == cp:
            return hp
    if len(pset) > 1:
        for ct, hp in pairs:
            if re.search(rf"\b{re.escape(ct)}\b", cp):
                return hp

    best_hp, best_score = None, 0
    for ct, hp in pairs:
        score = rfuzz.token_sort_ratio(cp, ct)
        if score > best_score:
            best_hp, best_score = hp, score
    return best_hp if best_score >= 80 else None


def parse_llm_mapping(resp_text: str, candidate_ids: set):
    try:
        js = json.loads(resp_text)
    except json.JSONDecodeError:
        js = None

    if isinstance(js, dict):
        candidate = next(
            (
                js[k].strip().strip('"')
                for k in ("hpo_id", "HPO_ID", "hp_id", "id")
                if isinstance(js.get(k), str)
            ),
            None,
        )
        if candidate:
            low = candidate.lower()
            if low in ("null", "no candidate fit"):
                return None, "null_label", js
            if candidate in candidate_ids:
                return candidate, "ok", js
            return candidate, "hp_not_in_candidates", js

    for m in re.findall(r"(HP:\d{6,7})", resp_text):
        if m in candidate_ids:
            return m, "regex_fallback", None

    return None, "no_hpo_found", None


def generate_hpo_terms(df_row: pd.DataFrame, system_message: str) -> pd.DataFrame:
    phrase = df_row["phrase"].iloc[0].strip()
    normalized = phrase.lower().replace("-", " ").strip()
    category = df_row["category"].iloc[0]
    original = df_row["original_sentence"].iloc[0]
    metadata_list = df_row["unique_metadata"].iloc[0] or []

    candidates = []
    seen = set()
    for m in metadata_list:
        term = m.get("phrase") or m.get("info")
        hp = m.get("hp_id")
        if term and hp and hp not in seen:
            candidates.append({"term": term, "id": hp})
            seen.add(hp)
    candidate_ids = {c["id"] for c in candidates}

    payload = json.dumps(
        {
            "phrase": normalized,
            "category": category,
            "original_sentence": original,
            "candidates": candidates,
        }
    )
    resp = llm_client.query(payload, system_message)
    hpo_id, reason, _ = parse_llm_mapping(resp, candidate_ids)

    if not hpo_id:
        cluster_idx = build_cluster_index(metadata_list)
        local_id = extract_hpo_term(normalized, metadata_list, cluster_idx)
        if local_id:
            hpo_id, reason = local_id, "fallback_local"

    return pd.DataFrame(
        [
            {
                "HPO_Terms": [{"phrase": phrase, "HPO_Term": hpo_id}],
                "raw_llm_resp": resp,
                "llm_parse_reason": reason,
            }
        ]
    )


# =========================== Input Validation ===========================

def validate_input(df):
    if "clinical_note" not in df.columns:
        raise KeyError("Missing required column: 'clinical_note'.")
    df = df.dropna(subset=["clinical_note"]).copy()
    df["clinical_note"] = df["clinical_note"].astype(str)
    df["clinical_note"] = df["clinical_note"].apply(clean_note)
    if "patient_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["patient_id"] = df.index + 1
    else:
        df["patient_id"] = df["patient_id"].astype(int)
    return df


# =========================== State Management ===========================

def load_state(temp_files):
    state = {}
    for key, path in temp_files.items():
        try:
            if os.path.exists(path):
                df = pd.read_pickle(path)
                state[key] = df
            else:
                state[key] = pd.DataFrame()
        except Exception as e:
            state[key] = pd.DataFrame()
            print(f"Warning loading '{key}': {e}. Starting fresh for this key.")
    return state


def save_state_checkpoint(state, temp_files, keys=("input", "combined", "exact", "non_exact", "final")):
    for key in keys:
        df = state.get(key)
        if df is None or df.empty:
            continue
        rel_path = temp_files.get(key)
        if not rel_path:
            logger.log(f"Warning: No path configured for '{key}'. Skipping.")
            continue
        abs_path = os.path.abspath(rel_path)
        tmp_path = abs_path + ".tmp"
        try:
            df.to_pickle(tmp_path)
            shutil.move(tmp_path, abs_path)
            logger.log(
                f"[SAVE] Checkpointed '{key}' ({len(df)} rows) -> {abs_path}", once=True
            )
        except Exception as e:
            logger.log(f"Error saving '{key}': {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# =========================== Output ===========================

def process_results(final_df):
    if final_df.empty:
        logger.log("No final results to process.")
        return

    choice = input("Save results as CSV or display? (save/display): ").strip().lower()
    if choice == "save":
        fname = input("Output CSV filename (e.g., results.csv): ").strip()
        if not fname:
            print("Filename cannot be empty. Skipping save.")
            return
        rows = []
        for idx, r in final_df.iterrows():
            pid = r.get("patient_id", idx)
            for term in r.get("HPO_Terms", []):
                ph = term.get("phrase", "").strip()
                cat = term.get("category", "")
                hp = term.get("HPO_Term") or ""
                if isinstance(hp, str):
                    hp = hp.replace("HP:HP:", "HP:")
                if hp:
                    rows.append(
                        {
                            "Patient ID": pid,
                            "Category": cat,
                            "Phenotype name": ph,
                            "HPO ID": hp,
                        }
                    )
                else:
                    print(
                        f"Warning: Blank HPO_Term for patient {pid}, "
                        f"phrase '{ph}', category '{cat}' - not included in CSV."
                    )
        if rows:
            output_df = pd.DataFrame(rows)
            output_df.to_csv(fname, index=False)
            logger.log(f"Saved tabular results to {fname}")
        else:
            logger.log("No valid HPO terms to save in tabular format.")
        final_df.to_csv(f"{os.path.splitext(fname)[0]}_json_raw.csv", index=False)
        logger.log(f"Saved raw JSON results to {os.path.splitext(fname)[0]}_json_raw.csv")

    elif choice == "display":
        tbl = []
        for idx, r in final_df.iterrows():
            pid = r.get("patient_id", idx)
            for term in r.get("HPO_Terms", []):
                ph = term.get("phrase", "").strip()
                cat = term.get("category", "")
                hp = term.get("HPO_Term") or ""
                if isinstance(hp, str):
                    hp = hp.replace("HP:HP:", "HP:")
                tbl.append(
                    {
                        "Case": f"Case {pid}",
                        "Category": cat,
                        "Phenotype name": ph,
                        "HPO ID": hp,
                    }
                )
        if tbl:
            print(tabulate(pd.DataFrame(tbl), headers="keys", tablefmt="psql"))
        else:
            logger.log("No terms to display.")
    else:
        print("Invalid choice; please enter 'save' or 'display'.")


def cleanup(temp_files, success):
    if success:
        logger.log("Pipeline succeeded. Cleaning up temporary files...")
        for path in temp_files.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                logger.log(f"Error removing temp file {path}: {e}")
    else:
        logger.log("Pipeline failed. Keeping temporary files for debugging/resume.")


# =========================== Main Pipeline ===========================

def main():
    parser = argparse.ArgumentParser(description="RAG-HPO: HPO term extraction from clinical notes")
    parser.add_argument("--csv", help="Path to CSV file with clinical notes")
    parser.add_argument("--api-key", help="LLM API key")
    parser.add_argument("--base-url", help="LLM API base URL")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--max-tokens", type=int, help="Max tokens per day")
    parser.add_argument("--max-qpm", type=int, help="Max queries per minute")
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--output", help="Output CSV filename (skips interactive prompt)")
    parser.add_argument("--output-mode", choices=["save", "display"], default=None,
                        help="Output mode (skips interactive prompt)")
    args = parser.parse_args()

    # 1) Initialize environment
    global llm_client
    llm_client = check_and_initialize(args)
    if not os.path.exists(FLAG_FILE):
        logger.log("Environment initialized.")

    logger.log("Starting HPO extraction pipeline...")
    start_time = time.time()

    # 2) Load checkpointed state
    state = load_state(TEMP_FILES)
    for key, df in state.items():
        if not df.empty:
            logger.log(f"Loaded checkpoint '{key}' ({len(df)} rows)")
        else:
            logger.log(f"No checkpoint found for '{key}', starting fresh.")

    # 3) Ingest notes on first run
    if state["combined"].empty:
        csv_path = args.csv
        if not csv_path:
            if input("Manual notes? (yes/no): ").strip().lower() == "yes":
                notes = []
                while True:
                    note = input("Note (or 'done'): ")
                    if note.lower() == "done":
                        break
                    notes.append(note)
                df_input = pd.DataFrame({"clinical_note": notes})
            else:
                while True:
                    fname = input("CSV filename: ")
                    try:
                        raw = pd.read_csv(fname)
                        df_input = validate_input(raw)
                        break
                    except Exception as e:
                        print(f"Error loading CSV: {e}")
        else:
            raw = pd.read_csv(csv_path)
            df_input = validate_input(raw)

        state["input"] = df_input
        save_state_checkpoint(state, TEMP_FILES, keys=["input"])
    else:
        logger.log("Resuming from existing 'combined' checkpoint.")

    # 4) Initialize models and indices
    emb_model = initialize_embeddings_model(use_sbert=True)
    docs, emb_matrix = load_vector_db(META_PATH, VEC_PATH)
    index = create_faiss_index(emb_matrix, metric="cosine")
    cluster_index = build_cluster_index(docs)

    success = False
    try:
        # 5) Process raw clinical notes
        if state["combined"].empty:
            logger.log("Processing clinical notes...")
            combined = pd.DataFrame()
            pids = sorted(state["input"]["patient_id"].unique())
            total = len(pids)
            for i, pid in enumerate(tqdm(pids, desc="Processing Notes", unit="note")):
                note = state["input"].loc[
                    state["input"]["patient_id"] == pid, "clinical_note"
                ].iloc[0]
                res = process_row(note, system_message_I, emb_model, index, docs)
                if not res.empty:
                    res["patient_id"] = pid
                    combined = pd.concat([combined, res], ignore_index=True)
                if (i + 1) % 5 == 0 or (i + 1) == total:
                    state["combined"] = combined.copy()
                    save_state_checkpoint(state, TEMP_FILES, keys=["combined"])
            state["combined"] = combined.copy()
            save_state_checkpoint(state, TEMP_FILES, keys=["combined"])
        else:
            logger.log("Skipped clinical processing; checkpoint exists.")

        # 6) Split exact vs non-exact
        if state["exact"].empty or state["non_exact"].empty:
            logger.log("Splitting exact vs non-exact...")
            df = state["combined"].copy()
            if "HPO_Term" not in df.columns:
                df["HPO_Term"] = np.nan
            if not df.empty:
                df["HPO_Term"] = df.apply(
                    lambda r: extract_hpo_term(r["phrase"], r["unique_metadata"], cluster_index)
                    if pd.isna(r["HPO_Term"])
                    else r["HPO_Term"],
                    axis=1,
                ).astype(object).where(lambda x: pd.notna(x), np.nan)

            exact_df, non_exact_df = split_exact_nonexact(df, hpo_term_col="HPO_Term")
            state["exact"] = exact_df
            state["non_exact"] = non_exact_df
            save_state_checkpoint(state, TEMP_FILES, keys=["exact", "non_exact"])
        else:
            logger.log("Skipped splitting; checkpoints exist.")

        # 7) Post-process non-exact entries
        non_ex = state["non_exact"].copy()
        for col in ("llm_parse_reason", "raw_llm_resp"):
            non_ex[col] = non_ex.get(col, pd.Series(dtype="object")).astype("object")

        idxs = non_ex[
            (non_ex["category"] == "Abnormal")
            & (non_ex["HPO_Term"].isna())
            & (non_ex["HPO_Term"] != "No Candidate Fit")
        ].index

        if not idxs.empty:
            logger.log(f"Generating HPO for {len(idxs)} entries...")
            counter = 0
            for i, idx in enumerate(tqdm(idxs, desc="Generating HPO", unit="entry")):
                row_df = non_ex.loc[[idx]]
                out_df = generate_hpo_terms(row_df, system_message_II)
                hp = out_df.at[0, "HPO_Terms"][0]["HPO_Term"] if not out_df.empty else None
                if "llm_parse_reason" in out_df.columns:
                    non_ex.at[idx, "llm_parse_reason"] = out_df.at[0, "llm_parse_reason"]
                if "raw_llm_resp" in out_df.columns:
                    non_ex.at[idx, "raw_llm_resp"] = out_df.at[0, "raw_llm_resp"]
                non_ex.at[idx, "HPO_Term"] = hp or "No Candidate Fit"

                counter += 1
                if counter >= 50 or (i + 1) == len(idxs):
                    state["non_exact"] = non_ex.copy()
                    save_state_checkpoint(state, TEMP_FILES, keys=["non_exact"])
                    counter = 0
        else:
            logger.log("No non-exact entries to process.")

        # 8) Compile final results
        if state["final"].empty:
            logger.log("Compiling final results...")
            merged = pd.concat([state["exact"], state["non_exact"]], ignore_index=True)
            merged = merged.dropna(subset=["HPO_Term"])
            if not merged.empty:
                grouped = (
                    merged.groupby("patient_id")[["phrase", "category", "HPO_Term"]]
                    .apply(lambda g: g.to_dict("records"))
                    .reset_index(name="HPO_Terms")
                )
                state["final"] = grouped.copy()
            else:
                state["final"] = pd.DataFrame(columns=["patient_id", "HPO_Terms"])
            save_state_checkpoint(state, TEMP_FILES, keys=["final"])
        else:
            logger.log("Skipped final compilation; checkpoint exists.")

        # 9) Output
        if args.output and args.output_mode:
            # Non-interactive output
            if args.output_mode == "save":
                rows = []
                for _, r in state["final"].iterrows():
                    pid = r.get("patient_id", _)
                    for term in r.get("HPO_Terms", []):
                        ph = term.get("phrase", "").strip()
                        cat = term.get("category", "")
                        hp = term.get("HPO_Term") or ""
                        if isinstance(hp, str):
                            hp = hp.replace("HP:HP:", "HP:")
                        if hp:
                            rows.append({
                                "Patient ID": pid,
                                "Category": cat,
                                "Phenotype name": ph,
                                "HPO ID": hp,
                            })
                if rows:
                    pd.DataFrame(rows).to_csv(args.output, index=False)
                    logger.log(f"Saved to {args.output}")
            else:
                print(tabulate(pd.DataFrame(state["final"]), headers="keys", tablefmt="psql"))
        else:
            process_results(state["final"])

        success = True
        logger.log(f"Pipeline completed in {time.time() - start_time:.2f}s")

    except Exception:
        logger.log(f"Pipeline error. Saving progress and exiting.")
        save_state_checkpoint(state, TEMP_FILES)
        traceback.print_exc()
    finally:
        cleanup(TEMP_FILES, success)


if __name__ == "__main__":
    # Load prompts at module level so main() can access them
    prompts = load_prompts()
    system_message_I = prompts.get("system_message_I", "")
    system_message_II = prompts.get("system_message_II", "")
    llm_client = None
    main()
