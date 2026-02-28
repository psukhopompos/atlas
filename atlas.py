#!/usr/bin/env python3
"""
atlas.py v0.1 — Cognitive Atlas Builder

Build a cognitive atlas from your behavioral traces. $0. Under 30 minutes.

    python atlas.py --traces ~/Downloads/twitter-archive/   # Twitter export
    python atlas.py --traces my_data.jsonl                   # generic JSONL
    python atlas.py --classify "some text" --atlas atlas.json

Pipeline: Load → Embed (Gemini, free) → UMAP → HDBSCAN → Portrait (Flash, free) → Output
"""

import argparse
import base64
import json
import os
import re
import signal
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Event

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _check_api_key():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.")
        print("  Get a free key at https://aistudio.google.com/apikey")
        print("  Then: export GOOGLE_API_KEY=your_key")
        print("  Or create a .env file with GOOGLE_API_KEY=your_key")
        sys.exit(1)


from google import genai

# ─── Config ──────────────────────────────────────────────────────────────────

VERSION = "0.1.0"
EMBED_MODEL = os.environ.get("ATLAS_EMBED_MODEL", "gemini-embedding-001")
PORTRAIT_MODEL = os.environ.get("ATLAS_PORTRAIT_MODEL", "gemini/gemini-3-flash-preview")
EMBED_BATCH_SIZE = 100
EMBED_MAX_WORKERS = 5
EMBED_DIM = 3072
BORDER_RATIO = 0.85
SAMPLES_PER_CLUSTER = 25
EMBED_TRUNCATE = 2000
ANNOTATE_MODEL = os.environ.get("ATLAS_ANNOTATE_MODEL", PORTRAIT_MODEL)
ANNOTATE_MAX_WORKERS = int(os.environ.get("ATLAS_ANNOTATE_WORKERS", "10"))

_shutdown = Event()

def _handle_sigint(signum, frame):
    if _shutdown.is_set():
        print("\n  FORCE QUIT")
        sys.exit(1)
    print("\n  SIGINT — finishing in-flight work, then saving...")
    _shutdown.set()

signal.signal(signal.SIGINT, _handle_sigint)


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_twitter_js(filepath):
    raw = Path(filepath).read_text(errors="ignore")
    idx = raw.index("=")
    return json.loads(raw[idx + 1:].strip())


def load_twitter_archive(archive_path):
    archive = Path(archive_path)
    data_dir = None
    if (archive / "data").is_dir():
        data_dir = archive / "data"
    elif (archive / "tweets.js").exists():
        data_dir = archive
    else:
        for d in archive.iterdir():
            if d.is_dir() and (d / "data").is_dir():
                data_dir = d / "data"
                break
    if data_dir is None:
        raise FileNotFoundError(f"Cannot find Twitter data dir in {archive}")

    user_id = None
    account_js = data_dir / "account.js"
    if account_js.exists():
        try:
            acct = _parse_twitter_js(account_js)
            user_id = acct[0]["account"]["accountId"]
        except Exception:
            pass

    traces = []

    tweets_js = data_dir / "tweets.js"
    if tweets_js.exists():
        data = _parse_twitter_js(tweets_js)
        for item in data:
            t = item.get("tweet", item)
            text = t.get("full_text", "").strip()
            if text.startswith("RT @"):
                continue
            if not text or len(text) < 3:
                continue
            stripped = re.sub(r'https?://\S+', '', text).strip()
            if not stripped or len(stripped) < 3:
                continue
            is_reply = t.get("in_reply_to_status_id") is not None
            if is_reply:
                mention_stripped = re.sub(r'@\w+', '', stripped).strip()
                if not mention_stripped or len(mention_stripped) < 3:
                    continue
            traces.append({
                "text": text,
                "source": "tweet_reply" if is_reply else "tweet",
                "created_at": t.get("created_at", ""),
            })

    like_js = data_dir / "like.js"
    if like_js.exists():
        data = _parse_twitter_js(like_js)
        for item in data:
            lk = item.get("like", item)
            text = lk.get("fullText", "").strip()
            if not text:
                continue
            if "suspended account" in text or "unable to view" in text:
                continue
            stripped = re.sub(r'https?://\S+', '', text).strip()
            if not stripped or len(stripped) < 3:
                continue
            traces.append({
                "text": text,
                "source": "like",
                "created_at": "",
            })

    note_js = data_dir / "note-tweet.js"
    if note_js.exists():
        data = _parse_twitter_js(note_js)
        for item in data:
            nt = item.get("noteTweet", item)
            text = nt.get("core", {}).get("text", "").strip()
            if not text:
                continue
            traces.append({
                "text": text,
                "source": "note_tweet",
                "created_at": nt.get("createdAt", ""),
            })

    dm_js = data_dir / "direct-messages.js"
    if dm_js.exists() and user_id:
        data = _parse_twitter_js(dm_js)
        for item in data:
            conv = item.get("dmConversation", item)
            messages = conv.get("messages", [])
            msg_creates = [m["messageCreate"] for m in messages if "messageCreate" in m]
            user_msgs = [m for m in msg_creates if m.get("senderId") == user_id]
            for m in user_msgs:
                txt = m.get("text", "").strip()
                stripped = re.sub(r'https?://\S+', '', txt).strip()
                if stripped and len(stripped) >= 5:
                    traces.append({
                        "text": txt,
                        "source": "dm",
                        "created_at": m.get("createdAt", ""),
                    })

    gdm_js = data_dir / "direct-messages-group.js"
    if gdm_js.exists() and user_id:
        data = _parse_twitter_js(gdm_js)
        for item in data:
            conv = item.get("dmConversation", item)
            messages = conv.get("messages", [])
            msg_creates = [m["messageCreate"] for m in messages if "messageCreate" in m]
            user_msgs = [m for m in msg_creates if m.get("senderId") == user_id]
            for m in user_msgs:
                txt = m.get("text", "").strip()
                stripped = re.sub(r'https?://\S+', '', txt).strip()
                if stripped and len(stripped) >= 5:
                    traces.append({
                        "text": txt,
                        "source": "group_dm",
                        "created_at": m.get("createdAt", ""),
                    })

    return traces


def load_jsonl(path):
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text or len(text) < 3:
                continue
            traces.append({
                "text": text,
                "source": obj.get("source", "unknown"),
                "created_at": obj.get("created_at", ""),
            })
    return traces


def auto_load(path):
    p = Path(path)
    if p.is_dir():
        if (p / "data" / "tweets.js").exists() or (p / "tweets.js").exists():
            return load_twitter_archive(p), "twitter"
        for d in p.iterdir():
            if d.is_dir() and (d / "data" / "tweets.js").exists():
                return load_twitter_archive(d), "twitter"
        raise FileNotFoundError(f"No Twitter archive or JSONL found in {p}")
    elif p.suffix in (".jsonl", ".json"):
        return load_jsonl(p), "jsonl"
    else:
        return load_jsonl(p), "jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION (--deep mode)
# ═══════════════════════════════════════════════════════════════════════════════

ANNOTATE_PROMPT = """\
You are a cognitive mineralogist. You receive a behavioral trace — something \
a mind produced or consumed — and you decompose it into its constituent \
cognitive elements, the way a spectrometer decomposes light into frequencies.

You do not interpret. You do not judge. You dissolve and name what you find.

Your output is a JSON object. Every field is free-form text — you name what \
you see with the precision of someone who knows that vague names destroy \
information. "philosophy" is vague. "epistemological anxiety about grounding \
claims without infinite regress" is precise.

The fields:

{
  "domains": [
    // What territories of knowledge, experience, or culture are present?
    // Name them with enough precision that two traces sharing a domain
    // would be recognizably about the same thing.
    // Typically 1-5 domains per trace.
  ],

  "tension": // What collides, contradicts, or creates friction here?
             // If nothing collides, what pulls? What is the gravitational
             // center of this trace? Name the tension, not the topic.
             // null if the trace is purely utilitarian with no tension.

  "register": // How does the language itself behave? Not what it says —
              // how it moves. Density, rhythm, formality, orality,
              // code-switching. Describe the linguistic texture as if
              // you were describing a material: rough, compressed,
              // liquid, brittle, layered.

  "energy": // What state is the nervous system in? Not emotion (which
            // is interpretation) but energy: velocity, temperature,
            // pressure. A trace can be high-velocity and cold.
            // Describe the energetic signature.

  "compression": // How much is packed into how little space?
                 // "maximum" = each word carries multiple loads.
                 // "expansive" = the thought breathes and wanders.
                 // "utilitarian" = no compression, pure function.
                 // Name the compression pattern.

  "switching": // Does the trace shift between languages, registers,
               // or cognitive modes? If yes, name the switch points
               // and what triggers each transition.
               // null if the trace is uniform throughout.

  "action": // What is the mind DOING here? Not thinking — doing.
            // Building? Recognizing? Reacting? Processing? Seeking?
            // Defending? Performing? Name the verb.
}

Respond with valid JSON only. No markdown. No commentary.\
"""


def _annotate_one(trace):
    """Annotate a single trace via LLM. Returns extraction dict or None."""
    import litellm

    parts = [f"[source: {trace['source']}]"]
    if trace.get("created_at"):
        parts.append(f"[when: {trace['created_at']}]")
    parts.append(f"\n{trace['text'][:4000]}")
    user_msg = "\n".join(parts)

    for attempt in range(5):
        if _shutdown.is_set():
            return None
        try:
            response = litellm.completion(
                model=ANNOTATE_MODEL,
                messages=[
                    {"role": "system", "content": ANNOTATE_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            raw_clean = re.sub(r'^```json\s*', '', raw)
            raw_clean = re.sub(r'\s*```$', '', raw_clean)
            try:
                return json.loads(raw_clean)
            except json.JSONDecodeError:
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                time.sleep(min(5 * (2 ** attempt), 60))
            elif attempt < 4:
                time.sleep(2 * (attempt + 1))
            else:
                return None
    return None


def _annotation_to_text(extraction):
    """Convert extraction dict to embeddable text."""
    if extraction is None:
        return ""
    if isinstance(extraction, list):
        extraction = extraction[0] if extraction else {}
    if not isinstance(extraction, dict):
        return ""
    parts = []
    domains = extraction.get("domains")
    if domains:
        if isinstance(domains, list):
            domains = ", ".join(str(d) for d in domains)
        parts.append(f"domains: {domains}")
    for field in ("tension", "register", "energy", "compression", "switching", "action"):
        val = extraction.get(field)
        if val and str(val).lower() not in ("null", "none", ""):
            parts.append(f"{field}: {val}")
    return "\n".join(parts) if parts else ""


def annotate_all(traces, output_dir):
    """Annotate all traces with cognitive extraction. Returns list of annotation dicts."""
    import litellm
    litellm.suppress_debug_info = True

    checkpoint = Path(output_dir) / "_annotations.jsonl"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    annotations = [None] * len(traces)

    done_indices = set()
    if checkpoint.exists():
        with open(checkpoint) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = obj["index"]
                if idx < len(annotations):
                    annotations[idx] = obj["extraction"]
                    done_indices.add(idx)
        print(f"  Resuming: {len(done_indices)} already annotated")

    remaining = [(i, traces[i]) for i in range(len(traces)) if i not in done_indices]
    if not remaining:
        print(f"  All {len(traces)} traces already annotated")
        return annotations

    print(f"  Annotating {len(remaining)} traces ({ANNOTATE_MODEL})...")
    print(f"  Workers: {ANNOTATE_MAX_WORKERS} · Ctrl+C to stop and resume later")

    t0 = time.time()
    done_count = 0
    errors = 0
    lock = Lock()

    def process(item):
        nonlocal done_count, errors
        idx, trace = item
        extraction = _annotate_one(trace)
        with lock:
            if extraction:
                annotations[idx] = extraction
                with open(checkpoint, "a") as f:
                    f.write(json.dumps({"index": idx, "extraction": extraction},
                                       ensure_ascii=False) + "\n")
                done_count += 1
            else:
                errors += 1
            total = done_count + errors
            if total % 100 == 0 and total > 0:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - total) / rate if rate > 0 else 0
                print(f"    [{done_count + len(done_indices)}/{len(traces)}] "
                      f"{rate:.1f}/s · ~{eta/60:.0f}m left · {errors} err")

    with ThreadPoolExecutor(max_workers=ANNOTATE_MAX_WORKERS) as executor:
        futures = []
        for item in remaining:
            if _shutdown.is_set():
                break
            futures.append(executor.submit(process, item))
        for f in futures:
            if _shutdown.is_set():
                break
            f.result()

    elapsed = time.time() - t0
    print(f"  {done_count} annotated in {elapsed/60:.1f}m "
          f"({done_count/max(elapsed,1):.1f}/s), {errors} errors")

    for i in range(len(annotations)):
        if annotations[i] is None:
            annotations[i] = {}

    if done_count + len(done_indices) >= len(traces) and checkpoint.exists():
        checkpoint.unlink()

    return annotations


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def _embed_batch(client, texts):
    for attempt in range(8):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
            )
            return [emb.values for emb in result.embeddings]
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                delay = min(5 * (2 ** attempt), 60)
                print(f"    Rate limited, waiting {delay}s...")
                time.sleep(delay)
                continue
            if attempt < 7:
                time.sleep(2 ** min(attempt, 4))
                continue
            raise


def embed_all(traces, output_dir, texts_override=None):
    texts = texts_override if texts_override else [t["text"][:EMBED_TRUNCATE] for t in traces]
    n = len(texts)
    checkpoint = Path(output_dir) / "_embed_checkpoint.npz"

    start_idx = 0
    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
    if checkpoint.exists():
        data = np.load(checkpoint)
        saved_count = int(data["done_count"])
        if saved_count <= n:
            embeddings[:saved_count] = data["embeddings"][:saved_count]
            start_idx = saved_count
            print(f"  Resuming from checkpoint: {start_idx}/{n}")

    if start_idx >= n:
        return embeddings

    client = genai.Client()
    remaining = n - start_idx
    t0 = time.time()
    done_count = 0
    lock = Lock()

    batches = []
    for i in range(start_idx, n, EMBED_BATCH_SIZE):
        batches.append((i, texts[i:i + EMBED_BATCH_SIZE]))

    def process_batch(batch_data):
        nonlocal done_count
        if _shutdown.is_set():
            return
        idx, batch_texts = batch_data
        result = _embed_batch(client, batch_texts)
        arr = np.array(result, dtype=np.float32)
        with lock:
            embeddings[idx:idx + len(batch_texts)] = arr
            done_count += len(batch_texts)
            total = start_idx + done_count
            if done_count % 500 < EMBED_BATCH_SIZE:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (remaining - done_count) / rate if rate > 0 else 0
                print(f"    [{total}/{n}] {rate:.0f}/s · ~{eta:.0f}s left")
            if done_count % 5000 < EMBED_BATCH_SIZE:
                np.savez_compressed(checkpoint, embeddings=embeddings,
                                    done_count=np.array(total))

    with ThreadPoolExecutor(max_workers=EMBED_MAX_WORKERS) as executor:
        list(executor.map(process_batch, batches))

    total = start_idx + done_count
    np.savez_compressed(checkpoint, embeddings=embeddings, done_count=np.array(total))
    elapsed = time.time() - t0
    print(f"  {done_count} embedded in {elapsed:.0f}s ({done_count/max(elapsed,1):.0f}/s)")

    if total >= n and checkpoint.exists():
        checkpoint.unlink()

    return embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

def auto_tune(n_traces):
    mcs = max(15, min(500, int(n_traces * 0.0045)))
    ms = max(5, mcs // 10)
    return mcs, ms


def cluster(embeddings, min_cluster_size=None):
    import umap
    import hdbscan

    n = len(embeddings)
    print(f"  UMAP: projecting {n} points to 2D...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=42, verbose=False,
    )
    coords_2d = reducer.fit_transform(embeddings).astype(np.float32)
    print(f"  UMAP done in {time.time() - t0:.0f}s")

    mcs, ms = auto_tune(n)
    if min_cluster_size is not None:
        mcs = min_cluster_size
        ms = max(5, mcs // 10)

    params = {"min_cluster_size": mcs, "min_samples": ms,
              "cluster_selection_method": "leaf"}
    print(f"  HDBSCAN: min_cluster_size={mcs}, min_samples={ms}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=ms,
        cluster_selection_epsilon=0.0,
        cluster_selection_method="leaf", metric="euclidean",
    )
    labels = clusterer.fit_predict(coords_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} regions, {n_noise} noise ({n_noise/n*100:.1f}%)")

    if n_noise > 0 and n_clusters > 0:
        from scipy.spatial.distance import cdist
        centroids_2d = np.array([
            coords_2d[labels == c].mean(axis=0) for c in range(n_clusters)
        ])
        noise_mask = labels == -1
        dists = cdist(coords_2d[noise_mask], centroids_2d)
        labels[noise_mask] = dists.argmin(axis=1)
        print(f"  Assigned {n_noise} noise points to nearest region")

    if n_clusters < 3:
        print(f"  WARNING: Only {n_clusters} regions. Try --min-cluster-size with a smaller value.")
    elif n_clusters > 50:
        print(f"  WARNING: {n_clusters} regions. Try --min-cluster-size with a larger value.")

    return labels, coords_2d, n_clusters, params


# ═══════════════════════════════════════════════════════════════════════════════
# PORTRAITS
# ═══════════════════════════════════════════════════════════════════════════════

PORTRAIT_PROMPT = """\
You are writing a cognitive atlas. You receive data about a cluster of behavioral \
traces — things a single mind produced or consumed. Your job is to write a portrait \
of this cluster: who inhabits this region of cognitive space?

Write in second person ("you"). Be precise, specific, and vivid. No generic platitudes. \
Name the exact tensions, obsessions, registers, and energies you observe. \
If the cluster has a contradiction, name it. If it has a signature move, name it.

Structure your response as valid JSON:
{
  "name": "2-5 word evocative name (not generic like 'Technical Discussion')",
  "portrait": "3-5 paragraphs describing who lives here and what they do",
  "signature": "One sentence capturing this region's essence",
  "verbs": ["3-5 action verbs that define this region"],
  "borders": "1-2 sentences: what this region is NOT (negative space)"
}

Write in a mix of English and the trace language if the traces show code-switching. \
Match the register of the data — if the traces are raw and vulgar, your portrait \
should have that texture. If they are dense and technical, match that.

Respond with valid JSON only.\
"""


def _analyze_cluster(traces, labels, cluster_id, annotations=None):
    mask = [i for i, l in enumerate(labels) if l == cluster_id]
    ct = [(i, traces[i]) for i in mask]
    sources = Counter(t["source"] for _, t in ct)
    source_pcts = {s: f"{c/len(ct)*100:.0f}%" for s, c in sources.most_common(5)}

    rng = np.random.RandomState(42 + cluster_id)
    by_source = defaultdict(list)
    for idx, t in ct:
        by_source[t["source"]].append((idx, t))

    sampled = []
    remaining = SAMPLES_PER_CLUSTER
    for src in sorted(by_source.keys()):
        src_items = by_source[src]
        n_take = max(1, int(len(src_items) / len(ct) * SAMPLES_PER_CLUSTER))
        n_take = min(n_take, remaining, len(src_items))
        indices = rng.choice(len(src_items), size=n_take, replace=False)
        for i in indices:
            sampled.append(src_items[i])
        remaining -= n_take
        if remaining <= 0:
            break
    if remaining > 0:
        sampled_set = {idx for idx, _ in sampled}
        pool = [(idx, t) for idx, t in ct if idx not in sampled_set]
        if pool:
            extra = rng.choice(len(pool), size=min(remaining, len(pool)), replace=False)
            for i in extra:
                sampled.append(pool[i])

    sample_data = []
    for idx, t in sampled:
        item = dict(t)
        if annotations and idx < len(annotations) and annotations[idx]:
            item["_annotation"] = annotations[idx]
        sample_data.append(item)

    return {"cluster_id": cluster_id, "size": len(ct),
            "sources": source_pcts, "samples": sample_data}


def write_portraits(traces, labels, n_clusters, annotations=None):
    import litellm
    litellm.suppress_debug_info = True

    portraits = []
    for cid in range(n_clusters):
        if _shutdown.is_set():
            break
        analysis = _analyze_cluster(traces, labels, cid, annotations=annotations)
        print(f"  [{cid+1}/{n_clusters}] {analysis['size']} traces...", end=" ", flush=True)

        parts = [f"Cluster {cid} — {analysis['size']} traces"]
        parts.append(f"Source distribution: {json.dumps(analysis['sources'])}")
        parts.append("\nSample traces:\n")
        for t in analysis["samples"]:
            parts.append(f"---")
            parts.append(f"[{t['source']}] {t.get('created_at', '')}")
            ann = t.get("_annotation")
            if ann:
                if ann.get("action"):
                    parts.append(f"action: {ann['action']}")
                if ann.get("domains"):
                    d = ann["domains"]
                    parts.append(f"domains: {', '.join(d) if isinstance(d, list) else d}")
                if ann.get("tension"):
                    parts.append(f"tension: {ann['tension']}")
                if ann.get("register"):
                    parts.append(f"register: {ann['register']}")
                if ann.get("energy"):
                    parts.append(f"energy: {ann['energy']}")
                sw = ann.get("switching")
                if sw and str(sw).lower() not in ("null", "none"):
                    parts.append(f"switching: {sw}")
            parts.append(f"text: {t['text'][:300]}")

        user_prompt = "\n".join(parts)
        portrait_data = {"name": f"Region {cid}", "portrait": "", "signature": "",
                         "verbs": [], "borders": ""}

        for attempt in range(3):
            try:
                response = litellm.completion(
                    model=PORTRAIT_MODEL,
                    messages=[
                        {"role": "system", "content": PORTRAIT_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                portrait_data = json.loads(raw)
                break
            except json.JSONDecodeError:
                raw_clean = re.sub(r'^```json\s*', '', raw)
                raw_clean = re.sub(r'\s*```$', '', raw_clean)
                try:
                    portrait_data = json.loads(raw_clean)
                    break
                except json.JSONDecodeError:
                    if attempt < 2:
                        time.sleep(3 * (attempt + 1))
                    else:
                        print(f"error: bad JSON")
            except Exception as e:
                if attempt < 2:
                    err = str(e)
                    if "429" in err or "rate" in err.lower():
                        time.sleep(15 * (attempt + 1))
                    else:
                        time.sleep(3 * (attempt + 1))
                    continue
                print(f"error: {str(e)[:60]}")

        name = portrait_data.get("name", f"Region {cid}")
        print(f'"{name}"')

        portraits.append({
            "id": cid, "name": name,
            "portrait": portrait_data.get("portrait", ""),
            "signature": portrait_data.get("signature", ""),
            "verbs": portrait_data.get("verbs", []),
            "borders": portrait_data.get("borders", ""),
            "size": analysis["size"],
            "fraction": analysis["size"] / len(traces),
            "sources": analysis["sources"],
        })
    return portraits


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_centroids(embeddings, labels, n_clusters):
    centroids = np.zeros((n_clusters, EMBED_DIM), dtype=np.float32)
    for cid in range(n_clusters):
        mask = labels == cid
        if mask.sum() > 0:
            centroids[cid] = embeddings[mask].mean(axis=0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return centroids / norms


def compute_calibration(embeddings, labels, centroids_normed):
    n = len(embeddings)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    norms[norms == 0] = 1
    sample_normed = sample / norms
    sims = sample_normed @ centroids_normed.T
    dists = 1.0 - sims
    novelties = dists.mean(axis=1)
    return {
        "novelty_mean": float(np.mean(novelties)),
        "novelty_p95": float(np.percentile(novelties, 95)),
        "novelty_max": float(np.max(novelties)),
        "border_ratio": BORDER_RATIO,
    }


def save_atlas_json(portraits, centroids_normed, calibration, metadata, output_dir):
    centroids_bytes = centroids_normed.astype(np.float32).tobytes()
    centroids_b64 = base64.b64encode(centroids_bytes).decode("ascii")

    atlas = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "regions": portraits,
        "calibration": calibration,
        "centroids_b64": centroids_b64,
    }

    path = Path(output_dir) / "atlas.json"
    with open(path, "w") as f:
        json.dump(atlas, f, indent=2, ensure_ascii=False)
    size_kb = path.stat().st_size / 1024
    print(f"  atlas.json ({size_kb:.0f} KB)")
    return path


def save_atlas_md(portraits, n_traces, output_dir):
    parts = ["# COGNITIVE ATLAS\n"]
    parts.append(f"*{n_traces:,} traces across {len(portraits)} regions. "
                 f"Generated by atlas.py v{VERSION}.*\n")
    parts.append("---\n")

    for p in sorted(portraits, key=lambda x: -x["size"]):
        parts.append(f"\n## Region {p['id']}: {p['name']}")
        parts.append(f"*{p['size']:,} traces ({p['fraction']*100:.1f}%) | "
                     f"Sources: {json.dumps(p['sources'])}*\n")
        if p.get("portrait"):
            parts.append(p["portrait"])
        if p.get("signature"):
            parts.append(f"\n**Signature:** {p['signature']}")
        if p.get("verbs"):
            parts.append(f"\n**Verbs:** {', '.join(p['verbs'])}")
        if p.get("borders"):
            parts.append(f"\n**Borders:** {p['borders']}")
        parts.append("\n---\n")

    path = Path(output_dir) / "atlas.md"
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print(f"  atlas.md")
    return path


def save_atlas_html(traces, coords_2d, labels, portraits, n_clusters, output_dir):
    import plotly.graph_objects as go

    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#FF8C94",
        "#A8E6CF", "#FFD93D", "#6C5CE7", "#FD79A8", "#00CEC9",
        "#E17055", "#0984E3", "#00B894", "#FDCB6E", "#E84393",
        "#636E72", "#2D3436", "#B2BEC3", "#DFE6E9", "#74B9FF",
        "#55EFC4", "#81ECEC", "#FAB1A0", "#FF7675", "#A29BFE",
        "#DCDDE1", "#7F8FA6", "#40739E", "#E1B12C", "#44BD32",
        "#C23616", "#0097E6", "#8C7AE6", "#6AB04C", "#EB4D4B",
    ]

    name_lookup = {p["id"]: p["name"] for p in portraits}
    fig = go.Figure()

    for cid in range(n_clusters):
        mask = [i for i, l in enumerate(labels) if l == cid]
        if not mask:
            continue
        name = name_lookup.get(cid, f"Region {cid}")
        color = colors[cid % len(colors)]

        hovers = []
        for i in mask:
            t = traces[i]
            preview = t["text"][:120].replace("\n", " ")
            hovers.append(f"<b>{name}</b><br>source: {t['source']}<br>---<br>{preview}")

        fig.add_trace(go.Scattergl(
            x=[coords_2d[i, 0] for i in mask],
            y=[coords_2d[i, 1] for i in mask],
            mode="markers",
            marker=dict(size=2.5, color=color, opacity=0.7),
            name=f"{name} ({len(mask)})",
            text=hovers, hoverinfo="text",
        ))

    fig.update_layout(
        title="COGNITIVE ATLAS",
        template="plotly_dark",
        width=1600, height=1000,
        legend=dict(font=dict(size=10), y=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    path = Path(output_dir) / "atlas.html"
    with open(path, "w") as f:
        f.write(fig.to_html(include_plotlyjs="cdn"))
    print(f"  atlas.html")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFY
# ═══════════════════════════════════════════════════════════════════════════════

def load_atlas_json(path):
    with open(path) as f:
        atlas = json.load(f)
    n_regions = atlas["metadata"]["n_regions"]
    dim = atlas["metadata"]["embedding_dim"]
    raw = base64.b64decode(atlas["centroids_b64"])
    centroids = np.frombuffer(raw, dtype=np.float32).reshape(n_regions, dim)
    name_lookup = {r["id"]: r["name"] for r in atlas["regions"]}
    return {
        "centroids": centroids,
        "name_lookup": name_lookup,
        "calibration": atlas["calibration"],
        "n_regions": n_regions,
        "regions": atlas["regions"],
    }


def classify_text(text, atlas, client):
    for attempt in range(5):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL, contents=[text[:EMBED_TRUNCATE]],
            )
            vec = np.array(result.embeddings[0].values, dtype=np.float32)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(min(3 * (2 ** attempt), 30))
                continue
            if attempt < 4:
                time.sleep(1)
                continue
            raise

    norm = np.linalg.norm(vec)
    vec_n = vec / (norm if norm > 0 else 1)

    sims = atlas["centroids"] @ vec_n
    dists = 1.0 - sims
    ranked = np.argsort(dists)

    top3 = []
    for i in range(min(3, len(ranked))):
        cid = int(ranked[i])
        top3.append({
            "id": cid,
            "name": atlas["name_lookup"].get(cid, f"Region {cid}"),
            "confidence": float(sims[cid]),
            "distance": float(dists[cid]),
        })

    d1, d2 = dists[ranked[0]], dists[ranked[1]]
    br = atlas["calibration"]["border_ratio"]
    is_border = (d1 / d2) > br if d2 > 0 else False

    novelty = float(dists.mean())
    cal = atlas["calibration"]
    if novelty < cal["novelty_p95"]:
        novelty_label = "familiar"
    elif novelty < cal["novelty_max"]:
        novelty_label = "edge"
    else:
        novelty_label = "alien"

    result = {
        "top3": top3, "is_border": is_border,
        "border_pair": (top3[0]["name"], top3[1]["name"]) if is_border else None,
        "novelty": novelty, "novelty_label": novelty_label,
    }

    preview = text[:80].replace("\n", " ")
    if len(text) > 80:
        preview += "..."
    lines = ["=" * 55]
    lines.append(f'INPUT: "{preview}"')
    lines.append("-" * 55)
    lines.append("TOP REGIONS:")
    for i, r in enumerate(result["top3"]):
        pad = " " * max(1, 35 - len(r["name"]))
        conf = f'{r["confidence"]:.2f} confidence' if i == 0 else f'{r["confidence"]:.2f}'
        lines.append(f"  {i+1}. {r['name']}{pad}({conf})")

    if is_border:
        lines.append(f"\nBORDER: Yes — {top3[0]['name']} <> {top3[1]['name']}")
    else:
        lines.append("\nBORDER: No")
    lines.append(f"NOVELTY: {novelty:.3f} ({novelty_label} — "
                 f"P95={cal['novelty_p95']:.3f}, max={cal['novelty_max']:.3f})")
    lines.append("=" * 55)

    return result, "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="atlas.py v0.1 — Cognitive Atlas Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python atlas.py --traces ~/Downloads/twitter-archive/\n"
               "  python atlas.py --traces data.jsonl --output ./my-atlas\n"
               "  python atlas.py --classify \"some text\" --atlas atlas.json\n")
    parser.add_argument("--traces", help="Input: Twitter archive dir or JSONL file")
    parser.add_argument("--classify", help="Classify text against an atlas")
    parser.add_argument("--atlas", default="atlas.json", help="Path to atlas.json")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--deep", action="store_true",
                        help="Annotate traces via LLM before embedding (higher fidelity)")
    parser.add_argument("--min-cluster-size", type=int, help="Override HDBSCAN")
    parser.add_argument("--no-portraits", action="store_true", help="Skip portraits")
    parser.add_argument("--no-viz", action="store_true", help="Skip HTML map")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--version", action="version", version=f"atlas.py v{VERSION}")

    args = parser.parse_args()

    _check_api_key()

    if args.classify:
        atlas = load_atlas_json(args.atlas)
        client = genai.Client()
        _, formatted = classify_text(args.classify, atlas, client)
        print(formatted)
        return

    if not args.traces:
        parser.print_help()
        return

    deep = args.deep
    n_steps = 6 if deep else 5
    step = 0

    mode_label = "DEEP" if deep else "RAW"
    print(f"atlas.py v{VERSION} — Cognitive Atlas Builder ({mode_label})")
    print("=" * 55)
    t_start = time.time()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    step += 1
    print(f"\n[{step}/{n_steps}] Loading traces...")
    traces, fmt = auto_load(args.traces)
    src_counts = Counter(t["source"] for t in traces)
    print(f"  {fmt} format: {len(traces):,} traces")
    for src, cnt in src_counts.most_common():
        print(f"    {src}: {cnt:,}")

    if len(traces) < 50:
        print(f"\n  ERROR: Need at least 50 traces, got {len(traces)}.")
        return

    seen = set()
    unique = []
    for t in traces:
        key = t["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(t)
    if len(unique) < len(traces):
        print(f"  Deduplicated: {len(traces)} → {len(unique)}")
        traces = unique

    # Annotate (deep mode only)
    embed_texts = None
    annotations = None
    if deep:
        step += 1
        print(f"\n[{step}/{n_steps}] Annotating traces ({ANNOTATE_MODEL})...")
        annotations = annotate_all(traces, out)
        embed_texts = []
        fallback_count = 0
        for i, ann in enumerate(annotations):
            ann_text = _annotation_to_text(ann)
            if ann_text:
                embed_texts.append(ann_text[:EMBED_TRUNCATE])
            else:
                embed_texts.append(traces[i]["text"][:EMBED_TRUNCATE])
                fallback_count += 1
        if fallback_count:
            print(f"  {fallback_count} traces fell back to raw text")

    # Embed
    step += 1
    embed_label = "annotations" if deep else "traces"
    print(f"\n[{step}/{n_steps}] Embedding {len(traces):,} {embed_label} ({EMBED_MODEL})...")
    embeddings = embed_all(traces, out, texts_override=embed_texts)

    # Cluster
    step += 1
    print(f"\n[{step}/{n_steps}] Clustering (UMAP + HDBSCAN)...")
    labels, coords_2d, n_clusters, cluster_params = cluster(
        embeddings, min_cluster_size=args.min_cluster_size
    )
    centroids_normed = compute_centroids(embeddings, labels, n_clusters)

    # Portraits
    step += 1
    if args.no_portraits:
        print(f"\n[{step}/{n_steps}] Skipping portraits (--no-portraits)")
        portraits = []
        for cid in range(n_clusters):
            mask = labels == cid
            size = int(mask.sum())
            sources = Counter(traces[i]["source"] for i in range(len(traces)) if labels[i] == cid)
            source_pcts = {s: f"{c/size*100:.0f}%" for s, c in sources.most_common(5)}
            portraits.append({
                "id": cid, "name": f"Region {cid}", "portrait": "",
                "signature": "", "verbs": [], "borders": "",
                "size": size, "fraction": size / len(traces),
                "sources": source_pcts,
            })
    else:
        print(f"\n[{step}/{n_steps}] Writing {n_clusters} portraits ({PORTRAIT_MODEL})...")
        portraits = write_portraits(traces, labels, n_clusters, annotations=annotations)

    # Save
    step += 1
    print(f"\n[{step}/{n_steps}] Saving outputs...")
    calibration = compute_calibration(embeddings, labels, centroids_normed)
    extraction_mode = "derived" if deep else "raw"
    metadata = {
        "n_traces": len(traces),
        "n_regions": n_clusters,
        "sources": dict(src_counts.most_common()),
        "embedding_model": EMBED_MODEL,
        "embedding_dim": EMBED_DIM,
        "clustering_params": cluster_params,
        "portrait_model": PORTRAIT_MODEL if not args.no_portraits else None,
        "annotate_model": ANNOTATE_MODEL if deep else None,
        "extraction_mode": extraction_mode,
    }
    save_atlas_json(portraits, centroids_normed, calibration, metadata, out)
    save_atlas_md(portraits, len(traces), out)

    if not args.no_viz:
        save_atlas_html(traces, coords_2d, labels, portraits, n_clusters, out)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} minutes.")
    print(f"  To classify: python atlas.py --classify \"your text\" --atlas {out}/atlas.json")


if __name__ == "__main__":
    main()
