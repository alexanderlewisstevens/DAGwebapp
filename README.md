# Causal Playground

An interactive Dash web app for exploring conditional independence and simple causal DAGs on tabular (categorical) data.

## Features
- Upload a CSV or use the built-in sample dataset loader.
- Auto-detect categorical variables.
- Chi-square CI tests with Cramér's V and slice-level summaries.
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
