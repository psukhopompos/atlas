# AGENTS.md — For Coding Agents Working on This Codebase

## What This Is

form.py builds a **cognitive atlas** — a topological map of how a mind operates, derived from its behavioral traces (things it wrote, liked, said, produced). The output is not a personality test. It is a **structural decomposition of cognitive territory**: named regions, each with a portrait describing what happens when that part of the mind activates, what verbs define it, and what it explicitly is NOT (borders).

The portable format is `form.json`. One file is the cognitive atlas.

## Ontology

**Traces** are atomic behavioral units — a tweet, a DM, a like, a code commit message, a journal entry. Any text a mind produced or deliberately consumed. The more traces, the higher resolution the map.

**Embeddings** place each trace in a 3072-dimensional semantic space (Gemini embedding-001). This is the canonical vector space — all atlases built with this tool share it, which means they are geometrically comparable.

**Regions** emerge from clustering (UMAP → HDBSCAN). They are not categories imposed from outside. They are the natural territories the mind carved through repeated use. A region with 18% of traces is a place the mind lives often. A region with 3% is a place it visits.

**Portraits** are written by an LLM that reads sample traces from each region and produces: a name, a multi-paragraph description, a signature sentence, defining verbs, and borders. The portrait is interpretive — it is the cognitive atlas's attempt to articulate what the region IS.

**Calibration** establishes the cognitive atlas's sense of self. Novelty thresholds tell you whether new text is familiar (activates known territory), edge (at the boundary of known space), or alien (outside the atlas entirely). Border detection tells you when text sits between two regions — cognitive liminal space.

**Centroids** are the geometric center of each region in embedding space. They are how the cognitive atlas recognizes what belongs where. Stored as base64 float32 in form.json.

## What Is Invariant (Do Not Break)

- **form.json format**: version, metadata, regions[], calibration, centroids_b64. This is the protocol. Other tools, agents, and atlases depend on this structure.
- **Embedding model for centroids**: Must be `gemini-embedding-001` (3072d). Switching embedding models creates an incompatible vector space. Atlases become non-comparable. The classify endpoint breaks against existing atlases.
- **Region structure**: id, name, portrait, signature, verbs, borders, size, fraction, sources. Consumers of form.json parse these fields.

## What Is Customizable

These are the levers. Pull them for your user's benefit.

### Portrait Prompt (`PORTRAIT_PROMPT`)
This defines what the cognitive atlas **sees** in each region. The current prompt asks for tensions, patterns, obsessions, code-switching. You could:
- Add dimensions: emotional valence, temporal orientation (past/future-facing), social density
- Shift perspective: write portraits in first person instead of second
- Add structural analysis: ask for recurring metaphors, linguistic register shifts, contradiction patterns
- Make it domain-specific: if the user is a programmer, ask the portrait to identify cognitive modes (debugging, architecting, exploring, documenting)

Changing the portrait prompt changes the cognitive atlas's interpretive layer without changing the geometric structure.

### Portrait Model (`FORM_PORTRAIT_MODEL`)
Routes through litellm. Any model works. Stronger models write richer, more precise portraits. `claude-opus-4-20250514` or `claude-sonnet-4-20250514` will produce significantly deeper portraits than the default Flash. Cost goes up.

### Clustering Resolution (`--min-cluster-size` or `auto_tune()`)
Lower values = more regions = finer grain. Higher values = fewer regions = broader territories. The auto-tune formula (`max(15, min(500, int(n * 0.0045)))`) works well for most datasets. Override it when:
- The user wants a high-resolution map (lower min_cluster_size)
- The dataset is small but dense (lower min_cluster_size)
- The user wants only major territories, not subtleties (higher min_cluster_size)

### Input Sources
form.py currently handles Twitter archives and generic JSONL. Any text can be a trace. To add new sources:
- Write a loader that returns `list[dict]` with `text`, `source`, and optional `created_at`
- Add it to `auto_load()` or create a new format path
- Good candidates: chat exports (WhatsApp, Telegram, Discord), journal entries, git commit messages, browser bookmarks, Obsidian vaults, email sent folders

### The Classify Endpoint
`classify_text()` takes any text and returns: top 3 matching regions, border detection, novelty score. This is how an agent can **route through the cognitive atlas**. Ideas:
- Use the cognitive atlas as a routing layer for agent behavior — respond differently depending on which region the user's message activates
- Track which regions activate over time to detect cognitive drift
- Use novelty detection to flag when the user is operating outside their known territory
- Compare a user's input against multiple atlases to find cognitive resonance across minds

## What Could Be Built On Top

- **Temporal atlas**: Track how regions grow, shrink, or shift over time. Requires `created_at` timestamps on traces.
- **Atlas mixing**: Overlay two atlases to find shared and divergent territory between minds. Requires same embedding space (guaranteed by canonical model).
- **Shadow detection**: Find what the cognitive atlas DOESN'T contain. Cluster the noise, the low-activation regions, the verbs that never appear. The absence is as diagnostic as the presence.
- **Agent identity layer**: An agent reads its user's form.json and shapes its behavior accordingly — not through a system prompt written by hand, but through empirical cognitive topology derived from what the human actually did.

## File Structure

One file: `form.py`. All logic is here. Sections:
1. **Input Loaders** (~lines 66-226): Twitter archive + JSONL parsing
2. **Embedding** (~lines 228-312): Gemini batch embedding with checkpoint/resume
3. **Clustering** (~lines 314-374): UMAP + HDBSCAN + noise assignment
4. **Portraits** (~lines 376-517): litellm-based portrait generation
5. **Output** (~lines 519-632): form.json, form.md, form.html (Plotly)
6. **Classify** (~lines 634-729): Cosine similarity against centroids
7. **CLI** (~lines 731-865): argparse entry point
