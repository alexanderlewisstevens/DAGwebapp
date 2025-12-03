# Causal Playground – Phase 1

Phase 1 delivers a minimal, modular Dash app for quick conditional independence exploration on tabular data.

## Features
- Upload a CSV and infer categorical columns (<= 20 unique values by default).
- Choose X, Y, and conditioning set Z; run chi-square tests per slice with Cramér's V.
- View contingency-derived plots for a selected slice (countplot).
- See a placeholder DAG built from dataset columns (networkx + matplotlib), ready for future editing.
- Edit a DAG interactively via the DAG Editor panel (dash-cytoscape) to add/remove nodes and edges, and map selected DAG nodes to CI X/Y/Z controls.
- Check a DAG vs Data consistency panel that compares graph-implied independencies (via d-separation) against empirical CI tests, with configurable alpha/max-independencies.
- Select a consistency-table row to drive CI X/Y/Z and highlight nodes in the DAG.
- Export/import DAG + settings (alpha/max-independencies) as JSON.
- After uploading a CSV, the DAG auto-populates with one node per column; you can add edges interactively.
- Runtime logging writes to `logs/app.log` (rotating file handler) and records dataset loading, CI runs, and DAG rendering events.

## Getting started
1. Create/activate a Python 3.11+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Dash app:
   ```bash
   python -m causal_playground.app_dash.app
   ```
   Then open the printed local URL in your browser.
4. Check runtime logs in `logs/app.log` for breadcrumbs on uploads, CI runs, and DAG rendering.

## Tests
Run the small sanity suite with:
```bash
pytest
```

## Project layout
```
causal_playground/
  core/            # Framework-agnostic logic
  app_dash/        # Dash glue (layout + callbacks)
  tests/           # Basic unit tests
requirements.txt
README.md
ARCHITECTURE.md
DEV_LOG.md
```

## Notes and next steps
- Core modules are importable outside Dash; plotting returns Matplotlib figures for reuse.
- - DAG rendering is static for now; the DAG Editor offers basic add/remove interactions via dash-cytoscape; future phases will tie DAG state to CI logic and add d-separation tooling.
- See `ARCHITECTURE.md` for a system overview and `DEV_LOG.md` for chronological changes.
- Logging details live in `core/logging_config.py`; log output is under `logs/app.log`.
