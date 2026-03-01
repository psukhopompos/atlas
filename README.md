# form.py

Build a cognitive atlas from your behavioral traces. **$0. Under 30 minutes.**

Your Twitter archive goes in. A map of your mind comes out — 10-30 named cognitive regions, each with a portrait of how you think in that territory.

```
python form.py --traces ~/Downloads/twitter-archive/
```

## What it does

```
Input (Twitter archive or JSONL)
  → Embed raw text (Gemini embedding-001, free)
  → Project to 2D (UMAP, local)
  → Cluster (HDBSCAN, auto-tuned, local)
  → Portrait each region (any LLM via litellm — default: Gemini Flash, free)
  → Output: form.json + form.md + form.html
```

Three outputs:
- **form.json** — portable cognitive atlas with embeddings, centroids, calibration data. This is the format.
- **form.md** — human-readable portraits of each cognitive region
- **form.html** — interactive 2D map (Plotly)

## Install

```bash
pip install google-genai litellm numpy hdbscan umap-learn plotly scipy python-dotenv
```

Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then:

```bash
export GOOGLE_API_KEY=your_key
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your_key

# Optional: override portrait model (any litellm-supported model)
FORM_PORTRAIT_MODEL=gemini/gemini-3-flash-preview   # default
# Examples:
# FORM_PORTRAIT_MODEL=anthropic/claude-sonnet-4-20250514
# FORM_PORTRAIT_MODEL=openai/gpt-4o
# FORM_PORTRAIT_MODEL=ollama/llama3

# Embedding model (Gemini only — canonical vector space for the format)
FORM_EMBED_MODEL=gemini-embedding-001
```

## Usage

### Build a cognitive atlas

```bash
# From a Twitter/X archive (download yours at x.com/settings → Your Account → Download)
python form.py --traces ~/Downloads/twitter-archive/

# From any JSONL file
python form.py --traces my_data.jsonl

# Skip portraits (faster, just clusters)
python form.py --traces data.jsonl --no-portraits

# Custom output directory
python form.py --traces data.jsonl --output ./my-atlas

# Deep mode: annotate each trace via LLM before embedding (higher fidelity)
python form.py --traces data.jsonl --deep
```

### Classify text against a cognitive atlas

```bash
python form.py --classify "some text" --form form.json
```

Returns the top 3 matching regions, border detection (text that sits between two regions), and novelty score (how alien the text is to your atlas).

### JSONL format

If you're not using a Twitter archive, provide a JSONL file:

```json
{"text": "your trace here", "source": "optional_label"}
{"text": "another trace", "source": "conversation"}
```

Each line is one behavioral trace — a thing you wrote, said, liked, or produced. The more traces, the higher resolution the cognitive atlas.

## Options

```
--traces PATH        Input: Twitter archive directory or JSONL file
--deep               Annotate traces via LLM before embedding (higher fidelity)
--classify TEXT      Classify text against an existing cognitive atlas
--form PATH          Path to form.json (default: form.json)
--output DIR         Output directory (default: .)
--min-cluster-size N Override auto-tuned HDBSCAN parameter
--no-portraits       Skip Flash portraits (unnamed regions)
--no-viz             Skip HTML map generation
--seed N             Random seed (default: 42)
--version            Show version
```

### Raw vs Deep mode

**Raw mode** (default): Embeds trace text directly. Fast, free, good enough for most use cases. Clusters by surface content and style.

**Deep mode** (`--deep`): Each trace is first decomposed by an LLM into cognitive elements (domains, tension, register, energy, compression, action), then those annotations are embedded. Clusters by cognitive signature rather than surface content. Higher fidelity, but slower (one LLM call per trace). Supports checkpoint/resume — Ctrl+C to stop, run again to continue.

## The form.json format

One file is the cognitive atlas. The format:

```json
{
  "version": "0.1.0",
  "created_at": "2026-02-25T...",
  "metadata": {
    "n_traces": 50393,
    "n_regions": 23,
    "sources": {"tweet": 8432, "like": 31209, "dm": 4102},
    "embedding_model": "gemini-embedding-001",
    "embedding_dim": 3072,
    "clustering_params": {"min_cluster_size": 226, "min_samples": 22},
    "portrait_model": "gemini-2.5-flash",
    "extraction_mode": "raw"
  },
  "regions": [
    {
      "id": 0,
      "name": "Region Name",
      "portrait": "2-3 paragraph portrait...",
      "signature": "One sentence essence.",
      "verbs": ["diagnosing", "naming", "probing"],
      "borders": "What this region is NOT.",
      "size": 1200,
      "fraction": 0.142,
      "sources": {"tweet": "45%", "like": "30%"}
    }
  ],
  "calibration": {
    "novelty_mean": 0.176,
    "novelty_p95": 0.212,
    "novelty_max": 0.293,
    "border_ratio": 0.85
  },
  "centroids_b64": "<base64 float32 array>"
}
```

`extraction_mode: "raw"` means traces were embedded as raw text. Centroids are base64-encoded float32, shape `(n_regions, 3072)`. A 20-region cognitive atlas is ~250KB.

## How it works

1. **Load** — Parses your Twitter archive (tweets, likes, DMs, note tweets) or generic JSONL. Filters retweets, URL-only posts, and near-duplicates.

2. **Embed** — Each trace is embedded via Gemini's `embedding-001` model (3072 dimensions, free tier). Batched in groups of 100 with retry/backoff. Checkpoints to disk so you can resume if interrupted.

3. **Cluster** — UMAP projects to 2D, HDBSCAN finds natural clusters. Parameters auto-tune based on trace count. Noise points get assigned to the nearest region.

4. **Portrait** — Gemini Flash writes a portrait of each region from 25 sample traces: a name, 2-3 paragraph description, signature sentence, defining verbs, and borders (what the region is NOT).

5. **Output** — Everything saved to form.json (portable), form.md (readable), and form.html (visual).

## What you need

- Python 3.9+
- A free [Gemini API key](https://aistudio.google.com/apikey)
- Your Twitter/X archive (or any JSONL of text traces)

50k traces takes ~25 minutes. 5k traces takes ~3 minutes. The bottleneck is embedding (API calls), not compute.

## License

MIT
