# Causal Playground Architecture (Phase 1.5)

## Overview
Causal Playground lets a user upload a CSV, pick variables X and Y, optionally set conditioning variables Z, run chi-square conditional independence tests, view slice-level plots, and see a placeholder DAG of the dataset columns. The core logic is framework-agnostic and wired into a Dash UI.

## Modules
- `core/data_io.py`  
  Loads CSV data into pandas and infers categorical columns (defaults to <= 20 unique values). Provides basic metadata (all columns, categorical vs. non-categorical) for UI population.

- `core/ci_engine.py`  
  Implements `conditional_ci_summary`, grouping the data by conditioning variables (or all data if none). For each slice it builds a contingency table, runs `scipy.stats.chi2_contingency`, and reports n, chi2, p-value, dof, and Cramér’s V. `contingency_details` returns counts plus joint/row/column probabilities for a slice.

- `core/plotting.py`  
  Pure Matplotlib/seaborn plot generators. `plot_slice_countplot` and `plot_slice_heatmap` accept a DataFrame slice with columns x and y and return figures (not displayed). Dash converts them to images for rendering.

- `core/dag_model.py`  
  Minimal networkx-based DAG utilities to create, add/remove nodes and edges, and render a static Matplotlib visualization. Currently used as a placeholder graph of dataset columns.

- `core/logging_config.py`  
  Provides `setup_logging` to configure a rotating file logger (`logs/app.log`) with a standard formatter. Enables shared logging across the app.

- `app_dash/app.py`  
  Dash entrypoint and glue. Layout includes CSV upload, X/Y/Z dropdowns, run-tests button, summary table, slice selector, plot area, and DAG image. Callbacks: parse uploaded CSV -> metadata + stores; recompute CI summary on demand; render selected slice plot; render DAG.

## Workflow Narrative
1. User uploads a CSV via Dash upload component.
2. The file is parsed to a DataFrame; metadata infers categorical columns to populate dropdowns.
3. User selects X, Y, and optional Z set; clicking “Run tests” triggers `conditional_ci_summary`, producing a summary table of chi-square results per slice.
4. User selects a slice (or default first slice); Dash filters the DataFrame to that group and renders a plot via `plot_slice_countplot`.
5. In parallel, a simple DAG is built from column names and rendered with networkx as a placeholder for future interactive editing.

## Logging
- The shared logger `"causal_playground"` is configured via `core/logging_config.py` with a rotating file handler at `logs/app.log` and a console stream handler.
- Key events logged: CSV loads (shape/columns/categorical inference), CI runs (x, y, conds, slice count), and DAG rendering (node/edge counts). Warnings are emitted when slices are empty or degenerate.

## DAG Editor
- `dash-cytoscape` powers an interactive DAG editor panel. DAG state is stored server-side and represented in networkx via helpers in `core/dag_model.py`.
- Conversion helpers bridge networkx and cytoscape elements (`dag_to_cytoscape_elements`, `cytoscape_elements_to_dag`), enabling add/remove node and edge actions through Dash callbacks.
- Logging captures DAG modifications (node/edge counts) for traceability.
- DAG selections can push values into the CI controls: selecting nodes and using the “Use selected as X/Y” or “Use selected as Z” actions sets the corresponding CI dropdown values (valid only if nodes exist in the uploaded dataset).
- When a CSV is uploaded and the DAG is empty, it is auto-populated with one node per column (no edges); user edits persist thereafter.

## DAG vs Data Consistency
- D-separation checks are provided via `dag_model.is_d_separated` (pgmpy-backed) and `dag_model.implied_independencies` to enumerate graph-implied independencies over selected nodes (unconditional and single-condition by default).
- `ci_engine.test_independencies` runs empirical CI tests for each implied independence and reports p-values and decisions.
- The Dash “DAG vs Data” panel triggers these computations and displays whether graph-implied independencies are supported by the data. Users can choose alpha and a cap on how many independencies to test; table rows can be selected to drive CI X/Y/Z and highlight nodes in the DAG.

## Persistence
- DAG and settings (alpha, max independencies) can be exported/imported as JSON. Serialization helpers live in `dag_model`; the Dash app wires buttons for download/upload and applies imported settings and DAG state with logging.
