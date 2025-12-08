"""Dash entrypoint for the Causal Playground Phase 1 app."""

from __future__ import annotations

import base64
import io
import logging
import json
from io import StringIO
from typing import List

import numpy as np

import pandas as pd
import dash
from dash import Dash, Input, Output, State, ALL, callback, dash_table, dcc, html, no_update, ctx
import dash_cytoscape as cyto
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

from causal_playground.core import ci_engine, data_io, dag_model
from causal_playground.core.logging_config import setup_logging

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#22d3ee", "#60a5fa", "#f59e0b", "#34d399", "#f472b6"]
PLOT_BG = "#0f172a"


def parse_contents(contents: str, filename: str | None) -> pd.DataFrame:
    """Decode uploaded CSV contents into a DataFrame."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    buffer = io.StringIO(decoded.decode("utf-8"))
    return data_io.load_csv(buffer)


CYBORG_THEME = "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/cyborg/bootstrap.min.css"

setup_logging()
logger = logging.getLogger("causal_playground")


def _dag_nodes_from_elements(elements: list[dict] | None) -> List[str]:
    """Extract node ids from cytoscape elements (nodes and any edge endpoints)."""
    nodes: set[str] = set()
    for el in elements or []:
        data = el.get("data", {})
        if "id" in data:
            nodes.add(data["id"])
        if "source" in data:
            nodes.add(data["source"])
        if "target" in data:
            nodes.add(data["target"])
    return sorted(nodes)


def _base_stylesheet():
    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": "#1f8efa",
                "color": "#0b1020",
                "font-size": "12px",
                "text-outline-color": "#e0f2fe",
                "text-outline-width": 2,
                "width": 38,
                "height": 38,
                "border-width": 2,
                "border-color": "#9ef4ff",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#7dd3fc",
                "line-color": "#7dd3fc",
                "arrow-scale": 1.5,
            },
        },
        {
            "selector": "node:selected",
            "style": {
                "border-color": "#fbbf24",
                "border-width": 3,
                "background-color": "#22d3ee",
                "text-outline-color": "#0f172a",
                "text-outline-width": 3,
                "shadow-blur": 12,
                "shadow-color": "#22d3ee",
                "shadow-opacity": 0.6,
                "shadow-offset-x": 0,
                "shadow-offset-y": 0,
            },
        },
        {
            "selector": "edge:selected",
            "style": {
                "line-color": "#fbbf24",
                "target-arrow-color": "#fbbf24",
                "width": 4,
            },
        },
    ]


def _generate_sample_df(name: str) -> pd.DataFrame | None:
    """Return a sample dataset by name."""
    if name == "starter_5vars":
        np.random.seed(0)
        N = 500
        A = np.random.binomial(1, 0.5, size=N)
        B = np.array([np.random.binomial(1, 0.2 if a == 0 else 0.8) for a in A])
        C = np.array([np.random.binomial(1, 0.1 if b == 0 else 0.7) for b in B])
        D = np.random.binomial(1, 0.5, size=N)
        E = np.array([np.random.binomial(1, 0.3 if a == 0 else 0.7) for a in A])
        df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E})
        logger.info("Loaded sample dataset 'starter_5vars' shape=%s", df.shape)
        return df
    if name == "chain_clean":
        np.random.seed(1)
        N = 200
        A = np.random.binomial(1, 0.5, size=N)
        B = A.copy()
        C = B.copy()
        df = pd.DataFrame({"A": A, "B": B, "C": C})
        logger.info("Loaded sample dataset 'chain_clean' shape=%s", df.shape)
        return df
    if name == "fork_clean":
        np.random.seed(2)
        N = 200
        B = np.random.binomial(1, 0.5, size=N)
        A = B.copy()
        C = B.copy()
        df = pd.DataFrame({"A": A, "B": B, "C": C})
        logger.info("Loaded sample dataset 'fork_clean' shape=%s", df.shape)
        return df
    if name == "collider_clean":
        np.random.seed(3)
        N = 200
        A = np.random.binomial(1, 0.5, size=N)
        B = np.random.binomial(1, 0.5, size=N)
        C = (A ^ B).astype(int)
        df = pd.DataFrame({"A": A, "B": B, "C": C})
        logger.info("Loaded sample dataset 'collider_clean' shape=%s", df.shape)
        return df
    if name == "independent":
        np.random.seed(4)
        N = 200
        A = np.random.binomial(1, 0.5, size=N)
        B = np.random.binomial(1, 0.5, size=N)
        C = np.random.binomial(1, 0.5, size=N)
        df = pd.DataFrame({"A": A, "B": B, "C": C})
        logger.info("Loaded sample dataset 'independent' shape=%s", df.shape)
        return df
    return None

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[CYBORG_THEME])
server = app.server

app.layout = html.Div(
    className="cp-root",
    children=[
        dcc.Store(id="ci-roles", data={"x": None, "y": None, "z": []}),
        html.Header(
            className="cp-header",
            children=[
                html.Div("Causal Playground", className="cp-logo"),
                html.Div("Causal DAG EDA", className="cp-tagline"),
                html.Div(
                    className="cp-header-actions",
                    children=[
                        html.A("What is this?", id="help-toggle", n_clicks=0, className="cp-link", href="#"),
                        dcc.Dropdown(
                            id="theme-toggle",
                            options=[{"label": "Light", "value": "light"}, {"label": "Dark", "value": "dark"}],
                            value="dark",
                            clearable=False,
                            className="cp-theme-toggle",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="help-panel",
            children=html.Div(
                dcc.Markdown(
                    """
**What this app does:** It lets you draw a causal DAG, test its implied independencies against the data, and run conditional independence (CI) tests for chosen X/Y/Z.

**Typical loop:** load data → auto-nodes appear → add edges → pick X/Y/Z (or click nodes) → run CI → check DAG vs Data to see which independencies the graph gets wrong.
"""
                ),
                className="cp-card",
            ),
            style={"display": "none"},
        ),
        html.Main(
            className="cp-main",
            children=[
                dcc.Store(id="data-store"),
                dcc.Store(id="dag-store", data=dag_model.dag_to_cytoscape_elements(dag_model.create_empty_dag())),
                dcc.Store(id="edge-source", data=None),
                html.Div(
                    className="cp-grid",
                    children=[
                        html.Div(
                            className="cp-card cp-card--full",
                            children=[
                                html.H3("Data & CI"),
                                html.Div(
                                    className="cp-inline-help",
                                    children=[
                                        html.Span(
                                            "Propose a DAG, test implied independencies, and refine it for your causal query."
                                        ),
                                        html.A("See full workflow →", href="#", className="cp-link"),
                                    ],
                                ),
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
                                    className="upload-box",
                                    multiple=False,
                                ),
                                html.Div(
                                    [
                                        html.Label("Try a sample dataset"),
                                        html.Div(
                                            [
                                                html.Button("Starter 5 vars", id="sample-btn-starter", n_clicks=0, className="secondary-btn"),
                                                html.Button("Chain", id="sample-btn-chain", n_clicks=0, className="secondary-btn"),
                                                html.Button("Fork", id="sample-btn-fork", n_clicks=0, className="secondary-btn"),
                                                html.Button("Collider", id="sample-btn-collider", n_clicks=0, className="secondary-btn"),
                                                html.Button("Independent", id="sample-btn-independent", n_clicks=0, className="secondary-btn"),
                                            ],
                                            className="button-row",
                                        ),
                                    ],
                                    className="row",
                                ),
                                html.Div(id="metadata-display", className="meta-text"),
                                html.Div(id="column-buttons", className="pill-row"),
                                html.Div(id="role-display", className="meta-text"),
                                html.Button("Run CI tests", id="run-tests", n_clicks=0, className="primary-btn"),
                                html.Div(id="summary-container"),
                                dash_table.DataTable(
                                    id="summary-table",
                                    columns=[],
                                    data=[],
                                    page_size=10,
                                    style_table={"overflowX": "auto"},
                                    style_data_conditional=[],
                                ),
                                html.Div(id="ci-stats-cards", className="card-row"),
                            ],
                        ),
                        html.Div(
                            className="cp-card cp-card--wide",
                            children=[
                                html.H3("Current CI slice"),
                                html.Div(
                                    [
                                        html.Label("Select slice"),
                                        dcc.Dropdown(id="slice-dropdown", options=[], value=None),
                                        dcc.Graph(id="plot-graph", className="plot-img"),
                                        html.Div(id="ci-graphs-container", className="plot-grid"),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(
                            className="cp-card cp-card--wide",
                            children=[
                                html.H3("DAG editor"),
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            [
                                                html.Label("Mode"),
                                                dcc.RadioItems(
                                                    id="dag-mode",
                                                    options=[
                                                        {"label": "Select", "value": "select"},
                                                        {"label": "Add edges", "value": "add"},
                                                    ],
                                                    value="select",
                                                    inline=True,
                                                ),
                                                html.Div(id="dag-status-message", className="meta-text"),
                                                html.Button("Remove selected", id="dag-remove-selected", n_clicks=0, className="secondary-btn"),
                                                html.Div(
                                                    [
                                                    html.Button("Use selected as X/Y", id="dag-use-xy", n_clicks=0, className="secondary-btn"),
                                                    html.Button("Use selected as Z", id="dag-use-z", n_clicks=0, className="secondary-btn"),
                                                    ],
                                                    className="row",
                                                ),
                                            ],
                                            className="col",
                                        ),
                                    ],
                                ),
                                cyto.Cytoscape(
                                    id="dag-cytoscape",
                                    layout={"name": "cose"},
                                    style={"width": "100%", "height": "420px", "border": "1px solid #1f2937"},
                                    elements=[],
                                    stylesheet=_base_stylesheet(),
                                    userZoomingEnabled=False,
                                    userPanningEnabled=False,
                                    autoungrabify=True,
                                ),
                                html.Div(
                                    [
                                        html.Button("Download DAG/settings", id="download-dag-settings-btn", n_clicks=0, className="secondary-btn"),
                                        dcc.Download(id="download-dag-settings"),
                                        dcc.Upload(
                                            id="upload-dag-settings",
                                            children=html.Button("Upload DAG/settings", className="secondary-btn"),
                                            multiple=False,
                                        ),
                                    ],
                                    className="row",
                                    style={"marginTop": "10px"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="cp-card cp-card--narrow",
                            children=[
                                html.H3("DAG vs data fit"),
                                html.Div(
                                    [
                                        html.Label("Alpha"),
                                        html.Div(id="alpha-display", className="meta-text"),
                                        dcc.Slider(
                                            id="alpha-slider",
                                            min=0.001,
                                            max=0.2,
                                            step=0.001,
                                            value=0.01,
                                            marks={0.001: "0.001", 0.005: "0.005", 0.01: "0.01", 0.02: "0.02", 0.05: "0.05", 0.1: "0.10", 0.2: "0.20"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            included=False,
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Max independencies to test", style={"marginRight": "8px"}),
                                                dcc.Input(
                                                    id="max-independencies",
                                                    type="number",
                                                    min=1,
                                                    max=500,
                                                    value=50,
                                                    style={"width": "120px", "marginRight": "12px"},
                                                ),
                                                dcc.Checklist(
                                                    id="edge-only-consistency",
                                                    options=[{"label": "Edge-only mode", "value": "edge_only"}],
                                                    value=[],
                                                    style={"marginRight": "12px"},
                                                ),
                                                html.Button("Check DAG vs data", id="consistency-btn", n_clicks=0, className="primary-btn"),
                                            ],
                                            className="row",
                                        ),
                                        html.Div(id="independencies-summary", className="meta-text", style={"marginTop": "6px"}),
                                    ],
                                ),
                                html.Div(id="dag-vs-data-stats", className="card-row"),
                                dash_table.DataTable(
                                    id="consistency-table",
                                    columns=[],
                                    data=[],
                                    page_size=10,
                                    row_selectable="single",
                                    style_data_conditional=[],
                                    style_table={"overflowX": "auto", "marginTop": "10px"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output("data-store", "data"),
    Output("metadata-display", "children"),
    Input("upload-data", "contents"),
    Input("sample-btn-starter", "n_clicks"),
    Input("sample-btn-chain", "n_clicks"),
    Input("sample-btn-fork", "n_clicks"),
    Input("sample-btn-collider", "n_clicks"),
    Input("sample-btn-independent", "n_clicks"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_data_load(upload_contents, starter_clicks, chain_clicks, fork_clicks, collider_clicks, independent_clicks, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (no_update,) * 2
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    df = None
    source = ""
    if trigger_id == "upload-data" and upload_contents is not None:
        df = parse_contents(upload_contents, filename)
        source = f"upload:{filename}"
    else:
        btn_map = {
            "sample-btn-starter": "starter_5vars",
            "sample-btn-chain": "chain_clean",
            "sample-btn-fork": "fork_clean",
            "sample-btn-collider": "collider_clean",
            "sample-btn-independent": "independent",
        }
        if trigger_id in btn_map:
            sample_value = btn_map[trigger_id]
            df = _generate_sample_df(sample_value)
            source = f"sample:{sample_value}"

    if df is None:
        return (no_update,) * 2

    metadata = data_io.basic_metadata(df)
    logger.info(
        "Data loaded from %s - shape=%s columns=%s categorical=%s",
        source,
        df.shape,
        list(df.columns),
        metadata["categorical_columns"],
    )
    data_json = df.to_json(date_format="iso", orient="split")
    meta_text = f"Loaded {len(df)} rows. Categorical columns: {', '.join(metadata['categorical_columns'])}"
    return data_json, meta_text


@callback(
    Output("column-buttons", "children"),
    Output("role-display", "children"),
    Input("data-store", "data"),
    Input("ci-roles", "data"),
)
def render_column_buttons(data_json, roles):
    roles = roles or {"x": None, "y": None, "z": []}
    if not data_json:
        return [], "Load data or choose a sample to set X/Y/Z."
    df = pd.read_json(StringIO(data_json), orient="split")
    cols = list(df.columns)
    buttons = []
    for col in cols:
        classes = ["pill-btn"]
        if roles.get("x") == col:
            classes.append("active-x")
        if roles.get("y") == col:
            classes.append("active-y")
        if col in (roles.get("z") or []):
            classes.append("active-z")
        buttons.append(
            html.Button(
                col,
                id={"type": "col-btn", "col": col},
                n_clicks=0,
                className=" ".join(classes),
            )
        )
    role_text = f"X: {roles.get('x') or '-'} | Y: {roles.get('y') or '-'} | Z: {', '.join(roles.get('z') or []) or '-'}"
    return buttons, role_text


@callback(
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Output("slice-dropdown", "options"),
    Output("slice-dropdown", "value"),
    Output("summary-table", "style_data_conditional"),
    Output("ci-stats-cards", "children"),
    Output("ci-graphs-container", "children"),
    Input("run-tests", "n_clicks"),
    Input("data-store", "data"),
    Input("ci-roles", "data"),
    prevent_initial_call=True,
)
def run_ci_tests(n_clicks, data_json, roles):
    roles = roles or {"x": None, "y": None, "z": []}
    x = roles.get("x")
    y = roles.get("y")
    z_values = roles.get("z") or []
    if not data_json or not x or not y or x == y:
        return [], [], [], None, [], [], []
    df = pd.read_json(StringIO(data_json), orient="split")
    df_cols = set(df.columns)
    if x not in df_cols or y not in df_cols:
        logger.warning("CI skipped - x or y not in dataset columns x=%s y=%s", x, y)
        return [], [], [], None, [], [], []
    conds: List[str] = z_values or []
    valid_conds = [c for c in conds if c in df_cols]
    dropped = set(conds) - set(valid_conds)
    if dropped:
        logger.warning("Dropping Z not in dataset columns: %s", list(dropped))
    logger.info("CI request received - x=%s y=%s conds=%s", x, y, valid_conds)
    summary_df = ci_engine.conditional_ci_summary(df, x, y, valid_conds)
    logger.info(
        "CI computed - x=%s y=%s conds=%s slices=%s",
        x,
        y,
        valid_conds,
        len(summary_df),
    )
    data_records = summary_df.to_dict(orient="records")
    slice_options = [
        {"label": f"{idx}: {row['condition']}", "value": idx} for idx, row in summary_df.iterrows()
    ]
    slice_value = slice_options[0]["value"] if slice_options else None
    # Style rows based on p-value for quick interpretation.
    style_data_conditional = [
        {"if": {"filter_query": "{p} < 0.001"}, "backgroundColor": "#ffebee"},
        {"if": {"filter_query": "{p} >= 0.001 && {p} < 0.01"}, "backgroundColor": "#ffe0b2"},
        {"if": {"filter_query": "{p} >= 0.01 && {p} < 0.05"}, "backgroundColor": "#fff3e0"},
        {"if": {"filter_query": "{p} >= 0.05"}, "backgroundColor": "#e8f5e9"},
    ]
    # Verdiction column
    for row in data_records:
        pval = row.get("p", 1)
        if pval < 0.001:
            row["verdict"] = "Strong dependence"
        elif pval < 0.01:
            row["verdict"] = "Dependence"
        elif pval < 0.05:
            row["verdict"] = "Weak dependence"
        else:
            row["verdict"] = "Looks independent"
    display_cols = [
        col
        for col in summary_df.columns
        if col not in {"contingency_counts", "contingency_probs", "x_levels", "y_levels"}
    ]
    display_records = []
    for row in data_records:
        display_row = {k: v for k, v in row.items() if k in display_cols or k == "verdict"}
        display_records.append(display_row)
    columns = [{"name": col, "id": col} for col in display_cols] + [{"name": "verdict", "id": "verdict"}]

    # Build stats cards
    cards = []
    if not summary_df.empty:
        row0 = summary_df.iloc[0]
        cards = [
            html.Div([html.Strong("chi2"), html.Div(f"{row0['chi2']:.3f}")], style={"padding": "8px", "border": "1px solid #eee"}),
            html.Div([html.Strong("p-value"), html.Div(f"{row0['p']:.3f}")], style={"padding": "8px", "border": "1px solid #eee"}),
            html.Div([html.Strong("Cramér's V"), html.Div(f"{row0['cramers_v']:.3f}")], style={"padding": "8px", "border": "1px solid #eee"}),
            html.Div([html.Strong("n"), html.Div(str(int(row0["n"])))], style={"padding": "8px", "border": "1px solid #eee"}),
            html.Div([html.Strong("dof"), html.Div(str(int(row0["dof"])))], style={"padding": "8px", "border": "1px solid #eee"}),
            html.Div([html.Strong("Verdict"), html.Div(data_records[0].get("verdict"))], style={"padding": "8px", "border": "1px solid #eee"}),
        ]

    # Compute global symmetric range for residuals across slices
    max_abs_residual = 0.0
    residuals_per_slice = []
    for _, row in summary_df.iterrows():
        probs = np.array(row.get("contingency_probs", []), dtype=float)
        if probs.size == 0:
            residuals_per_slice.append(None)
            continue
        row_marg = probs.sum(axis=1, keepdims=True)
        col_marg = probs.sum(axis=0, keepdims=True)
        expected = row_marg @ col_marg
        residual = probs - expected
        max_abs_residual = max(max_abs_residual, np.abs(residual).max())
        residuals_per_slice.append(residual)
    if max_abs_residual == 0:
        max_abs_residual = 1.0  # avoid zero-range

    # Build per-slice normalized heatmaps with residuals coloring (Plotly)
    graphs = []
    for idx, row in summary_df.iterrows():
        probs = row.get("contingency_probs", [])
        x_levels = row.get("x_levels", [])
        y_levels = row.get("y_levels", [])
        residual = residuals_per_slice[idx] if idx < len(residuals_per_slice) else None
        if probs and x_levels and y_levels and residual is not None:
            table = pd.DataFrame(probs, index=x_levels, columns=y_levels)
            residual_df = pd.DataFrame(residual, index=x_levels, columns=y_levels)
            p_slice = row.get("p", None)
            v_slice = row.get("cramers_v", None)
            badge = ""
            if p_slice is not None and v_slice is not None:
                if p_slice >= 0.05:
                    badge = f" | p={p_slice:.3f}, V={v_slice:.2f} → Indep-like"
                elif v_slice < 0.1:
                    badge = f" | p={p_slice:.3f}, V={v_slice:.2f} → Weak dep"
                else:
                    badge = f" | p={p_slice:.3f}, V={v_slice:.2f} → Strong dep"
            contrast_txt = ""
            if len(x_levels) == 2 and len(y_levels) == 2:
                try:
                    p_y1_x1 = table.iloc[1, 1] / table.iloc[1].sum() if table.iloc[1].sum() > 0 else 0
                    p_y1_x0 = table.iloc[0, 1] / table.iloc[0].sum() if table.iloc[0].sum() > 0 else 0
                    delta = p_y1_x1 - p_y1_x0
                    contrast_txt = f" | Δ P({y}={y_levels[1]} | {x}={x_levels[1]} vs {x_levels[0]}) = {delta:.2f}"
                except Exception:
                    pass
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=residual_df.values,
                        x=y_levels,
                        y=x_levels,
                        colorscale="RdBu",
                        zmin=-max_abs_residual,
                        zmax=max_abs_residual,
                        text=table.round(2).values,
                        texttemplate="%{text}",
                        showscale=False,
                    )
                ]
            )
            fig.update_layout(
                margin=dict(l=40, r=10, t=40, b=30),
                title=f"Slice: {row['condition']}{badge}{contrast_txt}",
                xaxis_title=y,
                yaxis_title=x,
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=PLOT_BG,
            )
            graphs.append(dcc.Graph(figure=fig, style={"width": "360px"}))

    return display_records, columns, slice_options, slice_value, style_data_conditional, cards, graphs


def _get_slice(df: pd.DataFrame, conds: List[str], slice_index: int | None) -> pd.DataFrame:
    if not conds:
        return df
    grouped = list(df.groupby(conds, dropna=False))
    if slice_index is None or slice_index >= len(grouped):
        return pd.DataFrame(columns=df.columns)
    return grouped[slice_index][1]


@callback(
    Output("plot-graph", "figure"),
    Input("slice-dropdown", "value"),
    State("summary-table", "data"),
    State("data-store", "data"),
    State("ci-roles", "data"),
)
def update_slice_plot(slice_index, summary_data, data_json, roles):
    roles = roles or {"x": None, "y": None, "z": []}
    x = roles.get("x")
    y = roles.get("y")
    z_values = roles.get("z") or []
    fig = go.Figure()
    if not data_json or not x or not y:
        fig.add_annotation(text="Select X and Y to plot", showarrow=False)
        fig.update_layout(template="plotly_dark", height=360)
        return fig
    df = pd.read_json(StringIO(data_json), orient="split")
    conds: List[str] = z_values or []
    sub_df = _get_slice(df, conds, slice_index)
    if sub_df.empty:
        fig.add_annotation(text="No data for selection", showarrow=False)
    else:
        fig = px.histogram(sub_df, x=x, color=y, barmode="group")
    fig.update_layout(
        height=360,
        margin=dict(l=40, r=10, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
    )
    return fig


@callback(Output("dag-cytoscape", "elements"), Input("dag-store", "data"))
def update_cytoscape_elements(store_data):
    return store_data or []


@callback(
    Output("dag-store", "data"),
    Output("edge-source", "data"),
    Output("dag-status-message", "children"),
    Input("upload-dag-settings", "contents"),
    Input("data-store", "data"),
    Input("dag-cytoscape", "tapNodeData"),
    Input("dag-remove-selected", "n_clicks"),
    State("dag-mode", "value"),
    State("edge-source", "data"),
    State("dag-cytoscape", "selectedNodeData"),
    State("dag-cytoscape", "selectedEdgeData"),
    State("dag-store", "data"),
    prevent_initial_call=True,
)
def update_dag_store(
    upload_contents,
    data_json,
    tap_node,
    remove_clicks,
    dag_mode,
    edge_source,
    selected_nodes,
    selected_edges,
    store_data,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    dag = dag_model.cytoscape_elements_to_dag(store_data or [])
    status = ""
    new_edge_source = edge_source

    if trigger_id == "upload-dag-settings" and upload_contents:
        try:
            content_type, content_string = upload_contents.split(",")
            decoded = base64.b64decode(content_string)
            payload = json.loads(decoded.decode("utf-8"))
            dag_payload = payload.get("dag", {})
            dag = dag_model.dag_from_serializable(dag_payload)
            logger.info(
                "Imported DAG/settings - nodes=%s edges=%s",
                dag.number_of_nodes(),
                dag.number_of_edges(),
            )
            status = f"Imported DAG/settings ({dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges)"
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to import DAG/settings: %s", exc)
            return no_update, edge_source, "Failed to import DAG/settings"
        return dag_model.dag_to_cytoscape_elements(dag), None, status

    if trigger_id == "data-store":
        if not data_json:
            return no_update, edge_source, status
        df = pd.read_json(StringIO(data_json), orient="split")
        dag = dag_model.dag_from_columns(list(df.columns))
        logger.info("Auto-populated DAG from CSV columns: %s", list(df.columns))
        return dag_model.dag_to_cytoscape_elements(dag), None, "DAG auto-populated from dataset columns"

    if trigger_id == "dag-remove-selected":
        if selected_nodes:
            node_ids = [node.get("id") for node in selected_nodes if node.get("id")]
            for node_id in node_ids:
                if dag.has_node(node_id):
                    dag_model.remove_node(dag, node_id)
            status = f"Removed nodes {node_ids}"
        elif selected_edges:
            for edge in selected_edges:
                src_edge = edge.get("source")
                dst_edge = edge.get("target")
                if src_edge and dst_edge and dag.has_edge(src_edge, dst_edge):
                    dag_model.remove_edge(dag, src_edge, dst_edge)
            status = "Removed selected edge(s)"
        new_edge_source = None
        return dag_model.dag_to_cytoscape_elements(dag), new_edge_source, status

    if trigger_id == "dag-cytoscape" and dag_mode == "add" and tap_node:
        node_id = tap_node.get("id")
        if not node_id:
            return no_update, new_edge_source, no_update
        if new_edge_source is None:
            new_edge_source = node_id
            status = f"Edge mode: source set to {node_id}. Click target."
            return no_update, new_edge_source, status
        if new_edge_source == node_id:
            status = "Cannot create self-loop; choose a different target."
            return no_update, new_edge_source, status
        if dag.has_edge(new_edge_source, node_id):
            status = "Edge already exists."
            new_edge_source = None
            return no_update, new_edge_source, status
        dag.add_edge(new_edge_source, node_id)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(new_edge_source, node_id)
            status = "Edge would create a cycle; skipped."
        else:
            status = f"Added edge {new_edge_source} → {node_id}"
        new_edge_source = None
        return dag_model.dag_to_cytoscape_elements(dag), new_edge_source, status

    if dag_mode != "add":
        new_edge_source = None

    return dag_model.dag_to_cytoscape_elements(dag), new_edge_source, status or no_update


@callback(
    Output("alpha-slider", "value"),
    Output("max-independencies", "value"),
    Input("upload-dag-settings", "contents"),
    State("alpha-slider", "value"),
    State("max-independencies", "value"),
    prevent_initial_call=True,
)
def load_settings_from_upload(upload_contents, cur_alpha, cur_max):
    if not upload_contents:
        return no_update, no_update
    try:
        content_type, content_string = upload_contents.split(",")
        decoded = base64.b64decode(content_string)
        payload = json.loads(decoded.decode("utf-8"))
        settings = payload.get("settings", {})
        alpha = settings.get("alpha", cur_alpha)
        max_ind = settings.get("max_independencies", cur_max)
        logger.info(
            "Imported settings - alpha=%s max_independencies=%s",
            alpha,
            max_ind,
        )
        return alpha, max_ind
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to import settings: %s", exc)
        return no_update, no_update


@callback(
    Output("consistency-table", "data"),
    Output("consistency-table", "columns"),
    Output("consistency-table", "style_data_conditional"),
    Output("dag-vs-data-stats", "children"),
    Output("independencies-summary", "children"),
    Input("consistency-btn", "n_clicks"),
    State("dag-store", "data"),
    State("data-store", "data"),
    State("alpha-slider", "value"),
    State("max-independencies", "value"),
    State("edge-only-consistency", "value"),
    prevent_initial_call=True,
)
def compute_dag_vs_data(n_clicks, dag_data, data_json, alpha_value, max_independencies, mode_value):
    if not data_json or dag_data is None:
        logger.warning("DAG vs Data skipped - missing data or DAG")
        return [], [], [], [], "Load data and a DAG to test implied independencies."
    df = pd.read_json(StringIO(data_json), orient="split")
    dag = dag_model.cytoscape_elements_to_dag(dag_data)
    valid_nodes = [n for n in dag.nodes if n in df.columns]
    if not valid_nodes:
        logger.warning("DAG vs Data skipped - no DAG nodes overlap dataset columns")
        return [], [], [], [], "DAG nodes do not overlap dataset columns."

    edge_only = "edge_only" in (mode_value or [])
    implied_all: list[tuple[str, str, tuple[str, ...]]] = []
    if edge_only:
        # Only test edges: parent-child conditional on other parents of child.
        for u, v in dag.edges:
            if u not in df.columns or v not in df.columns:
                continue
            conds = [p for p in dag.predecessors(v) if p != u and p in df.columns]
            implied_all.append((u, v, tuple(conds)))
    else:
        implied_all = dag_model.implied_independencies(dag, nodes=valid_nodes, max_conditions=1)

    if not implied_all:
        logger.info("DAG vs Data found no implied independencies to test")
        return [], [], [], [], "No implied independencies to test for this DAG/dataset."
    max_independencies = max_independencies or len(implied_all)
    implied = implied_all[: int(max_independencies)]
    alpha = alpha_value if alpha_value is not None else 0.01
    test_df = ci_engine.test_independencies(df, implied, alpha=alpha)
    if test_df.empty:
        return [], [], [], [], "No tests executed (empty result)."
    test_df["graph_says"] = "independent" if not edge_only else "edge_support"
    test_df["agreement"] = test_df["p_value"].apply(lambda p: "yes" if p >= alpha else "no")
    agreements = (test_df["agreement"] == "yes").sum()
    disagreements = (test_df["agreement"] == "no").sum()
    rate = agreements / len(test_df) if len(test_df) else 0.0
    if rate >= 0.95:
        fit_label = "Excellent fit"
    elif rate >= 0.8:
        fit_label = "Good fit"
    elif rate >= 0.6:
        fit_label = "Moderate fit"
    else:
        fit_label = "Poor fit"
    logger.info(
        "DAG vs Data consistency - mode=%s alpha=%s max_independencies=%s generated=%s tested=%s agreements=%s disagreements=%s",
        "edge_only" if edge_only else "implied",
        alpha,
        max_independencies,
        len(implied_all),
        len(test_df),
        agreements,
        disagreements,
    )
    columns = [{"name": col, "id": col} for col in test_df.columns]
    style_data_conditional = [
        {"if": {"filter_query": "{agreement} = 'no' && {p_value} < %f" % (alpha / 10)}, "backgroundColor": "#ffcdd2"},
        {"if": {"filter_query": "{agreement} = 'no'"}, "backgroundColor": "#ffebee"},
        {"if": {"filter_query": "{agreement} = 'yes'"}, "backgroundColor": "#e8f5e9"},
    ]
    stats_cards = [
        html.Div([html.Strong("Mode"), html.Div("Edges only" if edge_only else "Implied independencies")], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Independencies tested"), html.Div(str(len(test_df)))], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Agreements"), html.Div(str(int(agreements)))], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Disagreements"), html.Div(str(int(disagreements)))], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Agreement rate"), html.Div(f"{rate:.2f}")], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Fit"), html.Div(f"{fit_label} @ α={alpha:.3f}")], style={"padding": "8px", "border": "1px solid #eee"}),
    ]
    tested_label = "edge support checks" if edge_only else "implied independencies"
    summary_text = f"Testing {len(test_df)} of {len(implied_all)} {tested_label} (α={alpha:.3f}). Agreements={agreements}, disagreements={disagreements}."
    return test_df.to_dict("records"), columns, style_data_conditional, stats_cards, summary_text


def _edge_support_styles(dag_data, data_json, alpha):
    styles = []
    if not dag_data or not data_json:
        return styles
    dag = dag_model.cytoscape_elements_to_dag(dag_data)
    df = pd.read_json(StringIO(data_json), orient="split")
    for u, v in dag.edges:
        if u not in df.columns or v not in df.columns:
            continue
        table = pd.crosstab(df[u], df[v])
        if table.shape[0] == 0 or table.shape[1] == 0:
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        v_stat = ci_engine.cramers_v_from_table(table, chi2_stat=chi2)
        if p >= alpha:
            styles.append(
                {
                    "selector": f'[source = "{u}"][target = "{v}"]',
                    "style": {"line-color": "#b0bec5", "target-arrow-color": "#b0bec5", "line-style": "dashed"},
                }
            )
        elif v_stat > 0.3:
            styles.append(
                {
                    "selector": f'[source = "{u}"][target = "{v}"]',
                    "style": {"line-color": "#1565c0", "target-arrow-color": "#1565c0", "width": 3},
                }
            )
        elif v_stat > 0.1:
            styles.append(
                {
                    "selector": f'[source = "{u}"][target = "{v}"]',
                    "style": {"line-color": "#64b5f6", "target-arrow-color": "#64b5f6", "width": 2},
                }
            )
        else:
            styles.append(
                {
                    "selector": f'[source = "{u}"][target = "{v}"]',
                    "style": {"line-color": "#90caf9", "target-arrow-color": "#90caf9"},
                }
            )
    return styles


@callback(
    Output("dag-cytoscape", "stylesheet"),
    Input("consistency-table", "selected_rows"),
    State("consistency-table", "data"),
    State("dag-store", "data"),
    State("data-store", "data"),
    State("alpha-slider", "value"),
    State("ci-roles", "data"),
)
def build_dag_stylesheet(selected_rows, table_data, dag_data, data_json, alpha_value, roles):
    roles = roles or {"x": None, "y": None, "z": []}
    x_val = roles.get("x")
    y_val = roles.get("y")
    z_vals = roles.get("z") or []
    stylesheet = _base_stylesheet()
    stylesheet += _edge_support_styles(dag_data, data_json, alpha_value if alpha_value is not None else 0.01)
    # Role highlighting
    if x_val:
        stylesheet.append({"selector": f'[id = "{x_val}"]', "style": {"background-color": "#ffb74d"}})
    if y_val:
        stylesheet.append({"selector": f'[id = "{y_val}"]', "style": {"background-color": "#ff7043"}})
    for z in z_vals or []:
        stylesheet.append({"selector": f'[id = "{z}"]', "style": {"background-color": "#81c784"}})

    # Highlight from consistency table selection
    if table_data and selected_rows:
        row_index = selected_rows[0]
        if 0 <= row_index < len(table_data):
            row = table_data[row_index]
            x_sel = row.get("x")
            y_sel = row.get("y")
            conds = row.get("conds") or []
            if x_sel:
                stylesheet.append({"selector": f'[id = "{x_sel}"]', "style": {"border-color": "#ff9800", "border-width": 3}})
            if y_sel:
                stylesheet.append({"selector": f'[id = "{y_sel}"]', "style": {"border-color": "#ff9800", "border-width": 3}})
            for z in conds:
                stylesheet.append({"selector": f'[id = "{z}"]', "style": {"border-color": "#66bb6a", "border-width": 3}})
    return stylesheet


@callback(
    Output("alpha-display", "children"),
    Input("alpha-slider", "value"),
)
def show_alpha(alpha):
    if alpha is None:
        return ""
    return f"{alpha:.3f}"

@callback(
    Output("help-panel", "style"),
    Input("help-toggle", "n_clicks"),
    State("help-panel", "style"),
)
def toggle_help(n_clicks, current_style):
    display = (current_style or {}).get("display", "none")
    if n_clicks is None:
        return {"display": display}
    new_display = "block" if display == "none" else "none"
    return {"display": new_display}


@callback(
    Output("download-dag-settings", "data"),
    Input("download-dag-settings-btn", "n_clicks"),
    State("dag-store", "data"),
    State("alpha-slider", "value"),
    State("max-independencies", "value"),
    prevent_initial_call=True,
)
def download_dag_settings(n_clicks, dag_data, alpha_value, max_independencies):
    dag = dag_model.cytoscape_elements_to_dag(dag_data or [])
    payload = {
        "dag": dag_model.dag_to_serializable(dag),
        "settings": {
            "alpha": alpha_value if alpha_value is not None else 0.05,
            "max_independencies": max_independencies if max_independencies is not None else 50,
        },
    }
    logger.info(
        "Exported DAG/settings - nodes=%s edges=%s alpha=%s max_independencies=%s",
        dag.number_of_nodes(),
        dag.number_of_edges(),
        payload["settings"]["alpha"],
        payload["settings"]["max_independencies"],
    )
    return dict(content=json.dumps(payload, indent=2), filename="dag_settings.json")


@callback(
    Output("ci-roles", "data"),
    Input("data-store", "data"),
    Input({"type": "col-btn", "col": ALL}, "n_clicks"),
    Input("dag-use-xy", "n_clicks"),
    Input("dag-use-z", "n_clicks"),
    Input("dag-cytoscape", "tapNodeData"),
    Input("consistency-table", "selected_rows"),
    Input("dag-cytoscape", "selectedNodeData"),
    State({"type": "col-btn", "col": ALL}, "id"),
    State("consistency-table", "data"),
    State("ci-roles", "data"),
    State("data-store", "data"),
    State("dag-mode", "value"),
    prevent_initial_call=True,
)
def sync_roles(
    data_json_trigger,
    col_clicks,
    dag_xy_clicks,
    dag_z_clicks,
    tap_node,
    selected_rows,
    dag_selected_nodes,
    col_ids,
    consistency_data,
    roles,
    data_json_state,
    dag_mode,
):
    roles = roles or {"x": None, "y": None, "z": []}
    ctx_trigger = ctx.triggered_id

    if ctx_trigger == "data-store":
        return {"x": None, "y": None, "z": []}

    df_cols = []
    if data_json_state:
        df_cols = list(pd.read_json(StringIO(data_json_state), orient="split").columns)

    if isinstance(ctx_trigger, dict) and ctx_trigger.get("type") == "col-btn":
        col = ctx_trigger.get("col")
        if col not in df_cols:
            return roles
        # Toggle logic for buttons
        if roles.get("x") == col:
            roles["x"] = None
        elif roles.get("y") == col:
            roles["y"] = None
        elif col in (roles.get("z") or []):
            roles["z"] = [z for z in roles.get("z") or [] if z != col]
        else:
            if roles.get("x") is None:
                roles["x"] = col
            elif roles.get("y") is None:
                roles["y"] = col
            else:
                roles["z"] = (roles.get("z") or []) + [col]
        return roles

    if ctx_trigger == "dag-use-xy":
        selected_nodes = dag_selected_nodes
        if not data_json_state or not selected_nodes:
            logger.warning("DAG→CI X/Y mapping skipped - no data or selection")
            return roles
        selected_ids = [n.get("id") for n in selected_nodes if n.get("id")]
        valid = [n for n in selected_ids if n in df_cols]
        if len(valid) != 2:
            logger.warning("DAG→CI X/Y mapping requires exactly 2 nodes, got %s", len(valid))
            return roles
        roles["x"], roles["y"] = valid[0], valid[1]
        return roles

    if ctx_trigger == "dag-use-z":
        selected_nodes = dag_selected_nodes or []
        if not data_json_state:
            logger.warning("DAG→CI Z mapping skipped - no data available")
            return roles
        selected_ids = [n.get("id") for n in selected_nodes if n.get("id")]
        valid = [n for n in selected_ids if n in df_cols]
        roles["z"] = valid
        return roles

    if ctx_trigger == "dag-cytoscape":
        if dag_mode != "select" or not data_json_state or not tap_node:
            return roles
        node_id = tap_node.get("id")
        if not node_id or node_id not in df_cols:
            return roles
        if roles.get("x") == node_id and roles.get("y") == node_id:
            roles["x"], roles["y"] = roles["y"], roles["x"]
        elif roles.get("x") == node_id:
            roles["x"] = None
            roles["y"] = node_id
        elif roles.get("y") == node_id:
            roles["y"] = None
            roles["x"] = node_id
        elif roles.get("x") is None:
            roles["x"] = node_id
        elif roles.get("y") is None and node_id != roles.get("x"):
            roles["y"] = node_id
        else:
            z_list = roles.get("z") or []
            roles["z"] = [z for z in z_list if z != node_id] if node_id in z_list else z_list + [node_id]
        return roles

    if ctx_trigger == "consistency-table" and consistency_data and selected_rows:
        row_index = selected_rows[0]
        if 0 <= row_index < len(consistency_data):
            row = consistency_data[row_index]
            x_val = row.get("x")
            y_val = row.get("y")
            conds = row.get("conds") or []
            roles["x"] = x_val if x_val in df_cols else roles.get("x")
            roles["y"] = y_val if y_val in df_cols else roles.get("y")
            roles["z"] = [c for c in conds if c in df_cols]
        return roles

    return roles


def run_server(**kwargs):
    # Dash 3+ replaced run_server with run.
    app.run(**kwargs)


if __name__ == "__main__":
    run_server(debug=True)
