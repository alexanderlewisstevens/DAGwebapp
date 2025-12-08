## Causal DAG EDA workflow

### Purpose

This app is a focused “causal DAG EDA” tool. It helps you move from a vague story about how variables relate to a concrete causal DAG that:

- is broadly compatible with the observed data (its implied independencies mostly hold), and
- can answer specific causal questions (effects, adjustment sets, interventions).

The core loop is: propose a DAG → test it against the data → refine it for your causal query.

### Typical workflow

#### Step 0 – Basic data EDA

- Load a CSV or one of the bundled sample datasets.
- Inspect basic summaries: number of rows, which columns are treated as categorical, cardinalities, and missingness.
- Heed warnings about ultra-sparse or very high-cardinality variables; those are often poor candidates for CI tests.

#### Step 1 – Sketch an initial DAG

- The app can auto-populate one node per column.
- Use domain knowledge to add a small number of edges (simple patterns like chains, forks, colliders).
- Edges are added by switching to “Add edges” mode and clicking source then target; cycles and duplicates are blocked.

#### Step 2 – Explore raw associations (graph-agnostic)

Use the CI panel to study pairwise relationships between variables:

- choose X and Y, optionally condition on Z;
- inspect contingency tables, normalized probabilities, and CI test results (χ², p-value, Cramér’s V, n, dof, verdict).

Use this step to see which relationships clearly exist or clearly do not, before relying on the DAG.

#### Step 3 – Test implied independencies (graph-aware)

In the DAG vs Data panel, the app enumerates d-separation–implied independencies from the current DAG.

Each implied independence is tested via the CI engine; the table shows:

- variables (X, Y | Z),
- p-values and effect sizes,
- whether the independence is supported or violated at the current significance level α.

Summary cards report how many independencies were generated, tested, and how many agree with the data.

#### Step 4 – Diagnose violations

- Sort the DAG-vs-Data table by p-value or effect size to surface the strongest violations.
- Selecting a row will:
  - drive X/Y/Z in the CI panel,
  - highlight the involved nodes/edges in the DAG.

Use this view to ask:

- is a direct edge missing?
- is the direction wrong?
- are we conditioning on a collider or its descendant?

#### Step 5 – Answer a causal query (X → Y with adjustment Z)

- Use the DAG editor to encode the causal story you care about for a specific effect X → Y.
- Use X/Y/Z selection from the DAG to run CI tests aligned with your query.
- (Future) Check back-door identifiability, propose valid adjustment sets, and show CI results stratified by candidate Z sets.

#### Step 6 – Summarize DAG fit and robustness

The DAG vs Data panel provides a high-level scorecard:

- percentage of implied independencies supported,
- number and strength of violations,
- α-sensitivity (how results change as you move the alpha slider).

This gives a quick sense of whether the DAG is a reasonable description of the data for the question at hand.

### Implementation roadmap (dev-facing)

You already have pieces of this; turn these into issues/tasks.

#### A. Edge and role interaction

- Add “Select / Add edges” mode.
- In Add mode: click source then target; block cycles/duplicates; show status; visually mark the armed source.
- In Select mode: support multi-select and buttons:
  - “Set X from selection”
  - “Set Y from selection”
  - “Set Z from selection”
- Show X/Y/Z badges on nodes in the DAG (consistent colors across CI + DAG vs Data).

#### B. CI panel polish

- Stats cards: χ², p, Cramér’s V, n, dof, verdict.
- Verdict text + color coding (e.g., looks independent / weak evidence / clear dependence).
- Per-slice normalized heatmaps from contingency probabilities.
- Optional: tabs to toggle raw counts vs normalized tables.

#### C. DAG vs Data fit and explainability

- Summary cards: # generated, # tested, # agreements, # disagreements, agreement rate; fit label (Poor/Moderate/Good/Excellent).
- Explicit α helper text: “Tests with p < α are treated as violations.”
- Keep row coloring for agreement vs disagreement; optionally shade by p/effect size.
- Ensure selecting a row auto-runs CI for that triple (if not already) and updates DAG highlighting.

#### D. Edge support coloring

- Compute edge support via parent→child association (p, Cramér’s V).
- Thresholds/legend:
  - strong support → thick, saturated edge;
  - weak/ambiguous → normal;
  - unsupported (p > α, very low V) → dashed/faded.
- Keep styling consistent across panels.

#### E. Data EDA and relationships

- Basic summaries panel: missingness, cardinality, warnings on ultra-sparse/high-card variables.
- Optional “Relationships” tab:
  - Cramér’s V matrix for all pairs;
  - clicking a cell jumps into the CI panel for that pair.

#### F. Robustness / query-focused features (future)

- For a chosen query (X → Y):
  - check back-door identifiability;
  - list valid adjustment sets; mark them in DAG/CI UI.
- Add a compact “DAG scorecard” widget summarizing fit label, % supported, # strong violations, and identifiability status for the current query.

[2025-12-03 17:45] Causal DAG EDA workflow documentation
- Added a "Causal DAG EDA workflow" section to README describing the end-to-end causal DAG EDA loop (data EDA → sketch DAG → CI tests → DAG vs Data → query-focused analysis).
- Captured a dev-facing implementation roadmap (edge/role interaction, CI panel polish, DAG vs Data fit, edge support coloring, data EDA/relationships, robustness/query features) to drive upcoming issues and commits.
