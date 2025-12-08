[2025-12-03 14:50] Phase 1 implementation
- Implemented core utilities: data_io, ci_engine (chi-square + Cramér’s V), plotting, dag_model.
- Built Dash UI: CSV upload, X/Y/Z selection, CI summary, slice plotting, placeholder DAG.
- Added basic tests for CI and DAG.
- Next steps: logging, architecture doc, more tests.

[2025-12-03 14:58] Phase 1.5 scaffolding
- Added rotating file logging setup and integrated breadcrumbs in the Dash app.
- Documented architecture overview and updated README to reference logs/docs.
- Extended CI/DAG tests and added coverage for logging setup import.

[2025-12-03 15:25] Logging integration pass
- Verified app-level logging wiring and added breadcrumbs for CSV load, CI runs, and DAG rendering.
- Added logging in core modules (data_io, ci_engine, dag_model) for consistent high-level events.
- Updated README and ARCHITECTURE with logging notes.

[2025-12-03 15:35] Phase 2 kickoff - interactive DAG editor
- Added dash-cytoscape dependency and interactive DAG Editor panel in the Dash app.
- Extended dag_model with conversions between networkx DAGs and cytoscape elements, plus logging for DAG mutations.
- Updated tests for DAG conversions, README/ARCHITECTURE for the new editor, and maintained logging notes.

[2025-12-03 15:50] Phase 2 - DAG to CI wiring
- DAG Editor selections can now set CI X/Y/Z values via dedicated buttons; mappings only apply to nodes present in the uploaded dataset.
- Dropdown options reflect union of dataset categorical columns and DAG nodes; CI callbacks guard against invalid selections.
- Added logging for DAG→CI mappings and warnings when selections are invalid.
- Updated README/ARCHITECTURE to describe DAG-driven CI selection.

[2025-12-03 16:05] Phase 3 - DAG vs Data consistency
- Added d-separation helpers and implied independencies enumeration (pgmpy-backed) in `dag_model`.
- Introduced DAG vs Data panel to compute graph-implied independencies and test them via CI engine; results shown in-app.
- Added CI helper to test independencies, docs updated (README/ARCHITECTURE), and tests expanded for d-separation/implied independencies.

[2025-12-03 16:25] Phase 4 - DAG vs Data UX & persistence
- Added alpha slider and max-independencies control to the consistency panel; logs now capture these parameters.
- Consistency table row selection drives CI X/Y/Z and highlights nodes in the DAG without duplicate-output warnings.
- Implemented DAG/settings JSON export/import (nodes/edges + alpha/max-independencies) with logging; added serialization helpers and tests.

[2025-12-03 16:40] Auto-populate DAG from CSV columns
- Added helper to initialize a DAG with one node per dataset column.
- Uploading a CSV auto-populates the DAG if it is empty; existing DAG edits are preserved.
- Updated docs accordingly and added a unit test for the new helper.

[2025-12-03 16:55] QA checklist and smoke workflow test
- Added a README QA checklist covering upload → DAG auto-nodes → edge edits → DAG→CI mapping → CI runs → DAG vs Data → persistence.
- Added a smoke workflow test to exercise dag_from_columns, implied independencies, and CI independence tests on a tiny synthetic dataset.
- Standardised on `sample_data/starter_5vars.csv` as the shared demo dataset for internal QA and client demos.

[2025-12-03 17:05] Sample dataset loader
- Added in-app sample dataset controls (dropdown + button) to load starter_5vars without a local upload.
- Shared data-load callback now handles both uploads and sample loading.
[2025-12-03 17:10] Shared demo CSV and client docs
- Committed canonical demo CSV at `causal_playground/sample_data/starter_5vars.csv` (A→B→C, A→E, D noise).
- README restructured for clients: shared dataset, quick start, QA checklist, and usage flow spelled out.
[2025-12-03 17:20] Additional sample datasets
- Added more sample options (chain clean, fork clean, collider clean, independent) to the in-app loader and bundled CSVs under `sample_data/`.
- Sample generator covers these scenarios for quick demos (including noise-free cases).
[2025-12-03 17:35] DAG/CI UX polish and edge add mode
- CI panel: verdict column with color-coding, stats cards (χ², p, Cramér's V, n, dof), and per-slice normalized heatmaps from contingency probabilities.
- DAG styling: bezier edges with triangle arrowheads; edges styled by data support at current α; X/Y/Z roles highlighted.
- DAG vs Data: summary cards (tested, agreements, disagreements, rate) and clearer alpha slider with sparse marks/live value; table coloring.
- Added Select vs Add edge modes; in Add mode, clicking source then target adds an edge with cycle/duplicate checks and status messaging.
- Wrapped pd.read_json calls with StringIO to silence FutureWarnings; README/sample data in sync; tests remain green.
[2025-12-03 17:50] CI slice residual heatmaps
- CI slice plots now show residuals (p_ij - expected) on a diverging scale with consistent range across slices; annotations still display probabilities.
- Added per-slice badges (p, V, verdict) and binary contrast Δ for 2x2 cases when possible to make independence/dependence visually obvious.
[2025-12-03 18:10] Noise trim, stricter alpha, edge-only consistency
- Turned verbose d-separation/CI slice logs down to DEBUG to de-noise app.log.
- Raised the default alpha to 0.01 (higher bar for agreement) and added an edge-only consistency toggle plus fit badge in DAG vs Data.
- Added CI DAG context mini-view highlighting the current X/Y/Z on the graph; kept DAG vs Data computation manual (button-driven) to avoid recomputing on every edge edit.
[2025-12-03 18:40] UI shell & layout refresh
- Introduced a product-style shell: header with logo/tagline/help link, dark theme, gradient buttons, and card/grid layout for Data & CI, CI plots + DAG context, DAG editor, and DAG vs Data.
- Styled controls and cards consistently; centered content with max width; added inline help blurb under Data & CI; kept existing functionality and IDs intact.
- Added a backlog/roadmap section to README for future UX/feature work.
