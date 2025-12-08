# Causal Playground

An interactive Dash web app for exploring conditional independence and simple causal DAGs on tabular (categorical) data.

## Features
- Upload a CSV or use the built-in sample dataset loader.
- Auto-detect categorical variables.
- Chi-square CI tests with Cramér's V and slice-level summaries.
- Slice-by-slice residual heatmaps (deviation from independence baseline) with per-slice p/V badges and binary contrasts when applicable.
- Interactive DAG editor (dash-cytoscape): add/remove nodes and edges.
- DAG → CI mapping: select nodes in the DAG and push them into X/Y/Z.
- DAG vs Data panel: compare graph-implied independencies (d-separation) with empirical CI tests.
- Export/import DAG + settings (alpha/max-independencies) as JSON.
- Logging to `logs/app.log` for reproducible debugging.

## Shared demo dataset
- Paths under `causal_playground/sample_data/`
  - `starter_5vars.csv` (A → B → C, A → E, D independent noise)
  - `chain_clean.csv` (A → B → C, no noise)
  - `fork_clean.csv` (B → A and B → C, no noise)
  - `collider_clean.csv` (A → C ← B, no noise)
  - `independent.csv` (A, B, C independent)
- Canonical demos for screenshots, QA, and client exploration. These are also available via the in-app sample loader.

## Quick start (client)
1. Install and run:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python -m causal_playground.app_dash.app
   ```
   Open the URL shown in the terminal (typically http://127.0.0.1:8050).
2. Load the sample:
   - In the Data/CI section, choose “Starter 5 vars (A→B→C, A→E, D noise)” and click “Load sample dataset.”
   - X/Y/Z dropdowns list A–E; DAG shows nodes A–E (no edges yet).
3. Draw the DAG:
   - Add edges: A→B, B→C, A→E (leave D isolated).
4. Set CI variables via DAG:
   - Select A and C → “Use selected as X/Y.”
   - Select B → “Use selected as Z.”
   - Verify: X=A, Y=C, Z includes B.
5. Run CI:
   - Click “Run tests”; without Z expect strong A–C dependence; with Z=[B] expect weaker dependence (A ⟂ C | B).
6. DAG vs Data:
   - Set alpha (e.g., 0.05) and max independencies (e.g., 50); click “Compute.”
   - Inspect agreement; `(A, C | B)` should agree with the data.
7. Save/reload:
   - “Download DAG/settings” to export DAG and alpha/max.
   - “Upload DAG/settings” to restore later.

## QA checklist (internal)
- Load Starter 5 vars; confirm X/Y/Z populate and DAG auto-nodes appear.
- Add edges A→B, B→C, A→E; use DAG→CI to set X=A, Y=C, Z=[B].
- Run CI with/without Z; dependence weakens when conditioning.
- DAG vs Data shows `(A, C | B)` agreement.
- Export/import DAG + settings works.
- `logs/app.log` shows data load, DAG init/edits, DAG→CI mappings, CI runs, DAG vs Data summaries.

## Tests
Run the suite:
```bash
pytest
```
Includes unit tests for CI/DAG/d-separation, DAG vs Data helpers, and a smoke workflow test.

## Project layout
```
causal_playground/
  core/            # Framework-agnostic logic
  app_dash/        # Dash glue (layout + callbacks)
  tests/           # Unit and smoke tests
  sample_data/     # Bundled demo CSV
requirements.txt
README.md
ARCHITECTURE.md
DEV_LOG.md
```

## Notes
- Core modules are importable outside Dash; plotting returns Matplotlib figures for reuse.
- Logging config: `core/logging_config.py`; log output: `logs/app.log`.

## Backlog / Roadmap (high-value next steps)

- Layout & UX
  - Adopt a card/grid layout: Data & CI (full), CI plots (wide) + DAG context (narrow), DAG editor (wide), DAG vs Data (narrow).
  - Keep a consistent shell (header with help link), modern font, accent color, and better spacing/spacing hierarchy; ensure responsive behavior (tabs on small screens).
- DAG interaction
  - Refine role clicks: cycle node roles none → X → Y → Z → none; default X/Y to first categorical columns.
  - Lock zoom/pan annoyances; status text for edge add; add legend for X/Y/Z colors; highlight current CI path in the mini DAG.
- CI panel polish
  - Stronger stat cards (χ², p, V, n, dof, verdict with color).
  - Tabs/toggles for counts/probabilities/residuals; “last run” indicator/spinner; clear warnings for empty/large-Z slices.
- DAG vs Data fit
  - Scorecard: total implied independencies, tested, agreements/disagreements, agreement rate + fit label; make α rule explicit.
  - Edge-only mode clarified (local parent→child support with other parents conditioned); rows set X/Y/Z and scroll CI into view.
- Edge support styling
  - Normalize edge colors/weights across main DAG and context: strong/weak/unsupported with legend and thresholds.
- Data EDA & relationships
  - Add a “Data overview” card (rows, columns, categorical summary, missingness warnings).
  - Optional relationships tab: Cramér’s V matrix clickable to set X/Y.
- Causal query focus (later)
  - For chosen X→Y: back-door identifiability, candidate adjustment sets, one-click apply, and a small “query scorecard.”
- Advanced/latent nodes (later)
  - Optional latent node mode (no backing column, dashed outline), excluded from CI but used for d-separation/adjustment logic.
- Logging/stability
  - Reduce log noise (frequent DAG init/convert to DEBUG); keep one INFO line per DAG-vs-Data run.
  - Avoid redundant DAG initializations; keep a single authoritative DAG store.
