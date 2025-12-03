"""Dash entrypoint for the Causal Playground Phase 1 app."""

from __future__ import annotations

import base64
import io
import logging
import json
from io import StringIO
from typing import List

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dash
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update
import dash_cytoscape as cyto
import dash

from causal_playground.core import ci_engine, data_io, dag_model, plotting
from causal_playground.core.logging_config import setup_logging


def fig_to_base64(fig) -> str:
    """Convert a Matplotlib figure to a base64-encoded PNG data URL."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def parse_contents(contents: str, filename: str | None) -> pd.DataFrame:
    """Decode uploaded CSV contents into a DataFrame."""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    buffer = io.StringIO(decoded.decode("utf-8"))
    return data_io.load_csv(buffer)


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
        {"selector": "node", "style": {"label": "data(label)", "background-color": "#90caf9"}},
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#424242",
                "line-color": "#424242",
                "arrow-scale": 1.5,
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

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H1("Causal Playground – Phase 1"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and drop or ", html.A("select a CSV file")]),
            style={
                "width": "98%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(
            [
                html.Label("Or try a sample dataset"),
                dcc.Dropdown(
                    id="sample-dataset-dropdown",
                    options=[
                        {"label": "Starter 5 vars (A→B→C, A→E, D noise)", "value": "starter_5vars"},
                        {"label": "Chain clean (A→B→C, no noise)", "value": "chain_clean"},
                        {"label": "Fork clean (B→A, B→C, no noise)", "value": "fork_clean"},
                        {"label": "Collider clean (A→C←B, no noise)", "value": "collider_clean"},
                        {"label": "Independent (A,B,C independent)", "value": "independent"},
                    ],
                    placeholder="Choose a sample dataset",
                    style={"width": "50%"},
                ),
                html.Button("Load sample dataset", id="load-sample-btn", n_clicks=0, style={"marginLeft": "10px"}),
            ],
            style={"margin": "10px 0"},
        ),
        dcc.Store(id="data-store"),
        dcc.Store(id="dag-store", data=dag_model.dag_to_cytoscape_elements(dag_model.create_empty_dag())),
        html.Div(id="metadata-display"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Variable X"),
                        dcc.Dropdown(id="x-dropdown", options=[], value=None),
                    ],
                    style={"width": "30%", "display": "inline-block", "padding": "0 10px"},
                ),
                html.Div(
                    [
                        html.Label("Variable Y"),
                        dcc.Dropdown(id="y-dropdown", options=[], value=None),
                    ],
                    style={"width": "30%", "display": "inline-block", "padding": "0 10px"},
                ),
                html.Div(
                    [
                        html.Label("Conditioning variables Z"),
                        dcc.Dropdown(id="z-dropdown", options=[], value=[], multi=True),
                    ],
                    style={"width": "30%", "display": "inline-block", "padding": "0 10px"},
                ),
            ]
        ),
        html.Button("Run tests", id="run-tests", n_clicks=0, style={"margin": "10px 0"}),
        html.Div(id="summary-container"),
        dash_table.DataTable(
            id="summary-table",
            columns=[],
            data=[],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_data_conditional=[],
        ),
        html.Div(id="ci-stats-cards", style={"display": "flex", "gap": "10px", "marginTop": "10px"}),
    html.Hr(),
    html.Div(
        [
            html.Label("Select slice"),
            dcc.Dropdown(id="slice-dropdown", options=[], value=None),
            ],
            style={"width": "40%", "padding": "0 10px"},
        ),
        html.Div(
            [
                html.Div([html.Img(id="plot-image", style={"width": "100%", "maxWidth": "700px"})]),
                html.Div([html.Img(id="dag-image", style={"width": "100%", "maxWidth": "500px"})]),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "alignItems": "flex-start"},
        ),
        html.Div(id="ci-graphs-container", style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}),
        html.H2("DAG Editor"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Node name"),
                        dcc.Input(id="dag-node-input", type="text", placeholder="Node name"),
                        html.Button("Add node", id="dag-add-node", n_clicks=0, style={"marginLeft": "8px"}),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Add edge"),
                        dcc.Dropdown(id="dag-src-dropdown", placeholder="Source"),
                        dcc.Dropdown(id="dag-dst-dropdown", placeholder="Target", style={"marginTop": "6px"}),
                        html.Button("Add edge", id="dag-add-edge", n_clicks=0, style={"marginTop": "8px"}),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Button(
                    "Remove selected", id="dag-remove-selected", n_clicks=0, style={"marginBottom": "10px"}
                ),
                html.Div(
                    [
                        html.Button("Use selected as X/Y", id="dag-use-xy", n_clicks=0, style={"marginRight": "8px"}),
                        html.Button("Use selected as Z", id="dag-use-z", n_clicks=0),
                    ],
                    style={"marginBottom": "10px"},
                ),
                cyto.Cytoscape(
                    id="dag-cytoscape",
                    layout={"name": "cose"},
                    style={"width": "100%", "height": "400px", "border": "1px solid #ddd"},
                    elements=[],
                    stylesheet=_base_stylesheet(),
                ),
                html.Div(
                    [
                        html.Button("Download DAG/settings", id="download-dag-settings-btn", n_clicks=0),
                        dcc.Download(id="download-dag-settings"),
                        dcc.Upload(
                            id="upload-dag-settings",
                            children=html.Button("Upload DAG/settings"),
                            multiple=False,
                        ),
                    ],
                    style={"marginTop": "10px"},
                ),
            ]
        ),
        html.H2("DAG vs Data"),
        html.Div(
            [
        html.Label("Alpha"),
        html.Div(id="alpha-display", style={"marginLeft": "8px"}),
        dcc.Slider(
            id="alpha-slider",
            min=0.001,
            max=0.2,
            step=0.001,
            value=0.05,
            marks={0.01: "0.01", 0.02: "0.02", 0.05: "0.05", 0.1: "0.10", 0.2: "0.20"},
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
                html.Button("Compute DAG vs Data consistency", id="consistency-btn", n_clicks=0),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "10px"},
        ),
            ],
            style={"marginBottom": "10px"},
        ),
        html.Div(id="dag-vs-data-stats", style={"display": "flex", "gap": "10px", "marginBottom": "10px"}),
dash_table.DataTable(
    id="consistency-table",
            columns=[],
            data=[],
            page_size=10,
            row_selectable="single",
            style_data_conditional=[],
            style_table={"overflowX": "auto", "marginTop": "10px"},
        ),
    ]
)


@callback(
    Output("data-store", "data"),
    Output("metadata-display", "children"),
    Input("upload-data", "contents"),
    Input("load-sample-btn", "n_clicks"),
    State("upload-data", "filename"),
    State("sample-dataset-dropdown", "value"),
    prevent_initial_call=True,
)
def handle_data_load(upload_contents, sample_clicks, filename, sample_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (no_update,) * 2
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    df = None
    source = ""
    if trigger_id == "upload-data" and upload_contents is not None:
        df = parse_contents(upload_contents, filename)
        source = f"upload:{filename}"
    elif trigger_id == "load-sample-btn" and sample_value:
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
    Output("x-dropdown", "options"),
    Output("y-dropdown", "options"),
    Output("z-dropdown", "options"),
    Input("data-store", "data"),
    Input("dag-store", "data"),
    State("x-dropdown", "value"),
    State("y-dropdown", "value"),
)
def update_dropdown_options(data_json, dag_data, x_value, y_value):
    if not data_json and not dag_data:
        return [], [], []
    df_cols: List[str] = []
    if data_json:
        df = pd.read_json(StringIO(data_json), orient="split")
        df_cols = data_io.get_categorical_columns(df)
    dag_nodes = _dag_nodes_from_elements(dag_data)
    option_values = sorted(set(df_cols) | set(dag_nodes))

    x_options = [{"label": c, "value": c} for c in option_values]
    y_options = [{"label": c, "value": c} for c in option_values if c != x_value]
    z_blocklist = {x_value, y_value}
    z_options = [{"label": c, "value": c} for c in option_values if c not in z_blocklist]
    return x_options, y_options, z_options


@callback(
    Output("summary-table", "data"),
    Output("summary-table", "columns"),
    Output("slice-dropdown", "options"),
    Output("slice-dropdown", "value"),
    Output("summary-table", "style_data_conditional"),
    Output("ci-stats-cards", "children"),
    Output("ci-graphs-container", "children"),
    Input("run-tests", "n_clicks"),
    State("data-store", "data"),
    State("x-dropdown", "value"),
    State("y-dropdown", "value"),
    State("z-dropdown", "value"),
    prevent_initial_call=True,
)
def run_ci_tests(n_clicks, data_json, x, y, z_values):
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
    columns = [{"name": col, "id": col} for col in summary_df.columns]
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

    # Build per-slice normalized heatmaps
    graphs = []
    for idx, row in summary_df.iterrows():
        probs = row.get("contingency_probs", [])
        x_levels = row.get("x_levels", [])
        y_levels = row.get("y_levels", [])
        if probs and x_levels and y_levels:
            table = pd.DataFrame(probs, index=x_levels, columns=y_levels)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(table, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
            ax.set_title(f"Slice: {row['condition']}")
            ax.set_xlabel(y)
            ax.set_ylabel(x)
            fig.tight_layout()
            graphs.append(html.Img(src=fig_to_base64(fig), style={"width": "320px"}))

    return data_records, columns, slice_options, slice_value, style_data_conditional, cards, graphs


def _get_slice(df: pd.DataFrame, conds: List[str], slice_index: int | None) -> pd.DataFrame:
    if not conds:
        return df
    grouped = list(df.groupby(conds, dropna=False))
    if slice_index is None or slice_index >= len(grouped):
        return pd.DataFrame(columns=df.columns)
    return grouped[slice_index][1]


@callback(
    Output("plot-image", "src"),
    Input("slice-dropdown", "value"),
    State("summary-table", "data"),
    State("data-store", "data"),
    State("x-dropdown", "value"),
    State("y-dropdown", "value"),
    State("z-dropdown", "value"),
)
def update_slice_plot(slice_index, summary_data, data_json, x, y, z_values):
    if not data_json or not x or not y:
        return None
    df = pd.read_json(StringIO(data_json), orient="split")
    conds: List[str] = z_values or []
    sub_df = _get_slice(df, conds, slice_index)
    fig = plotting.plot_slice_countplot(sub_df, x, y)
    return fig_to_base64(fig)


@callback(
    Output("dag-image", "src"),
    Input("data-store", "data"),
)
def update_dag(data_json):
    dag = dag_model.create_empty_dag()
    if data_json:
        df = pd.read_json(StringIO(data_json), orient="split")
        for col in df.columns:
            dag_model.add_node(dag, col)
        # Simple chain placeholder to make a visible structure.
        for i in range(len(df.columns) - 1):
            dag_model.add_edge(dag, df.columns[i], df.columns[i + 1])
    logger.info("DAG rendered - nodes=%s edges=%s", dag.number_of_nodes(), dag.number_of_edges())
    fig = dag_model.dag_to_figure(dag)
    return fig_to_base64(fig)


@callback(Output("dag-cytoscape", "elements"), Input("dag-store", "data"))
def update_cytoscape_elements(store_data):
    return store_data or []


@callback(
    Output("dag-src-dropdown", "options"),
    Output("dag-dst-dropdown", "options"),
    Input("dag-store", "data"),
)
def update_dag_dropdowns(store_data):
    dag = dag_model.cytoscape_elements_to_dag(store_data or [])
    options = [{"label": n, "value": n} for n in dag.nodes]
    return options, options


@callback(
    Output("dag-store", "data"),
    Input("dag-add-node", "n_clicks"),
    Input("dag-add-edge", "n_clicks"),
    Input("dag-remove-selected", "n_clicks"),
    Input("upload-dag-settings", "contents"),
    Input("data-store", "data"),
    State("dag-node-input", "value"),
    State("dag-src-dropdown", "value"),
    State("dag-dst-dropdown", "value"),
    State("dag-cytoscape", "selectedNodeData"),
    State("dag-cytoscape", "selectedEdgeData"),
    State("dag-store", "data"),
    prevent_initial_call=True,
)
def update_dag_store(
    add_node_clicks,
    add_edge_clicks,
    remove_clicks,
    upload_contents,
    data_json,
    node_name,
    src,
    dst,
    selected_nodes,
    selected_edges,
    store_data,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    dag = dag_model.cytoscape_elements_to_dag(store_data or [])

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
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to import DAG/settings: %s", exc)
            return no_update
        return dag_model.dag_to_cytoscape_elements(dag)

    if trigger_id == "data-store":
        if not data_json:
            return no_update
        dag_current = dag_model.cytoscape_elements_to_dag(store_data or [])
        if dag_current.number_of_nodes() > 0 or dag_current.number_of_edges() > 0:
            logger.info("CSV uploaded but DAG already edited; keeping existing DAG")
            return no_update
        df = pd.read_json(StringIO(data_json), orient="split")
        dag = dag_model.dag_from_columns(list(df.columns))
        logger.info("Auto-populated DAG from CSV columns: %s", list(df.columns))
        return dag_model.dag_to_cytoscape_elements(dag)

    if trigger_id == "dag-add-node":
        if not node_name:
            return no_update
        dag_model.add_node(dag, node_name)
    elif trigger_id == "dag-add-edge":
        if not src or not dst or src == dst:
            return no_update
        dag_model.add_edge(dag, src, dst)
    elif trigger_id == "dag-remove-selected":
        if selected_nodes:
            node_ids = [node.get("id") for node in selected_nodes if node.get("id")]
            for node_id in node_ids:
                if dag.has_node(node_id):
                    dag_model.remove_node(dag, node_id)
        elif selected_edges:
            for edge in selected_edges:
                src_edge = edge.get("source")
                dst_edge = edge.get("target")
                if src_edge and dst_edge and dag.has_edge(src_edge, dst_edge):
                    dag_model.remove_edge(dag, src_edge, dst_edge)
    return dag_model.dag_to_cytoscape_elements(dag)


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
    Input("consistency-btn", "n_clicks"),
    State("data-store", "data"),
    State("dag-store", "data"),
    State("alpha-slider", "value"),
    State("max-independencies", "value"),
    prevent_initial_call=True,
)
def compute_dag_vs_data(n_clicks, data_json, dag_data, alpha_value, max_independencies):
    if not data_json or dag_data is None:
        logger.warning("DAG vs Data skipped - missing data or DAG")
        return [], [], [], []
    df = pd.read_json(StringIO(data_json), orient="split")
    dag = dag_model.cytoscape_elements_to_dag(dag_data)
    valid_nodes = [n for n in dag.nodes if n in df.columns]
    if not valid_nodes:
        logger.warning("DAG vs Data skipped - no DAG nodes overlap dataset columns")
        return [], [], [], []
    implied_all = dag_model.implied_independencies(dag, nodes=valid_nodes, max_conditions=1)
    if not implied_all:
        logger.info("DAG vs Data found no implied independencies to test")
        return [], [], [], []
    max_independencies = max_independencies or len(implied_all)
    implied = implied_all[: int(max_independencies)]
    alpha = alpha_value if alpha_value is not None else 0.05
    test_df = ci_engine.test_independencies(df, implied, alpha=alpha)
    if test_df.empty:
        return [], [], [], []
    test_df["graph_says"] = "independent"
    test_df["agreement"] = test_df["p_value"].apply(lambda p: "yes" if p >= alpha else "no")
    agreements = (test_df["agreement"] == "yes").sum()
    disagreements = (test_df["agreement"] == "no").sum()
    logger.info(
        "DAG vs Data consistency - alpha=%s max_independencies=%s generated=%s tested=%s agreements=%s disagreements=%s",
        alpha,
        max_independencies,
        len(implied_all),
        len(test_df),
        agreements,
        disagreements,
    )
    columns = [{"name": col, "id": col} for col in test_df.columns]
    style_data_conditional = [
        {"if": {"filter_query": "{agreement} = 'no'"}, "backgroundColor": "#ffebee"},
        {"if": {"filter_query": "{agreement} = 'yes'"}, "backgroundColor": "#e8f5e9"},
    ]
    stats_cards = [
        html.Div([html.Strong("Independencies tested"), html.Div(str(len(test_df)))], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Agreements"), html.Div(str(int(agreements)))], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Disagreements"), html.Div(str(int(disagreements)))], style={"padding": "8px", "border": "1px solid #eee"}),
        html.Div([html.Strong("Agreement rate"), html.Div(f"{(agreements/len(test_df)) if len(test_df) else 0:.2f}")], style={"padding": "8px", "border": "1px solid #eee"}),
    ]
    return test_df.to_dict("records"), columns, style_data_conditional, stats_cards


@callback(
    Output("dag-cytoscape", "stylesheet"),
    Input("consistency-table", "selected_rows"),
    State("consistency-table", "data"),
)
def highlight_dag_selection(selected_rows, table_data):
    stylesheet = _base_stylesheet()
    if not table_data or not selected_rows:
        return stylesheet
    row_index = selected_rows[0]
    if row_index < 0 or row_index >= len(table_data):
        return stylesheet
    row = table_data[row_index]
    x_val = row.get("x")
    y_val = row.get("y")
    conds = row.get("conds") or []
    highlight_nodes = []
    if x_val:
        highlight_nodes.append({"selector": f'[id = "{x_val}"]', "style": {"background-color": "#ffb74d"}})
    if y_val:
        highlight_nodes.append({"selector": f'[id = "{y_val}"]', "style": {"background-color": "#ffb74d"}})
    for z in conds:
        highlight_nodes.append({"selector": f'[id = "{z}"]', "style": {"background-color": "#81c784"}})
    return stylesheet + highlight_nodes


@callback(
    Output("alpha-display", "children"),
    Input("alpha-slider", "value"),
)
def show_alpha(alpha):
    if alpha is None:
        return ""
    return f"{alpha:.3f}"


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
    Output("x-dropdown", "value"),
    Output("y-dropdown", "value"),
    Output("z-dropdown", "value"),
    Input("data-store", "data"),
    Input("dag-use-xy", "n_clicks"),
    Input("dag-use-z", "n_clicks"),
    Input("consistency-table", "selected_rows"),
    State("consistency-table", "data"),
    State("x-dropdown", "value"),
    State("y-dropdown", "value"),
    State("z-dropdown", "value"),
    State("dag-cytoscape", "selectedNodeData"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def sync_ci_values(
    data_json_trigger,
    dag_xy_clicks,
    dag_z_clicks,
    selected_rows,
    consistency_data,
    cur_x,
    cur_y,
    cur_z,
    selected_nodes,
    data_json_state,
):
    """Single owner for CI dropdown values to avoid duplicate-output warnings."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return cur_x, cur_y, cur_z
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Reset on data upload
    if trigger_id == "data-store":
        return None, None, []

    if trigger_id == "dag-use-xy":
        if not data_json_state or not selected_nodes:
            logger.warning("DAG→CI X/Y mapping skipped - no data or selection")
            return no_update, no_update, cur_z
        selected_ids = [n.get("id") for n in selected_nodes if n.get("id")]
        if len(selected_ids) != 2:
            logger.warning("DAG→CI X/Y mapping requires exactly 2 nodes, got %s", len(selected_ids))
            return no_update, no_update, cur_z
        df = pd.read_json(StringIO(data_json_state), orient="split")
        df_cols = set(df.columns)
        valid = [n for n in selected_ids if n in df_cols]
        if len(valid) != 2:
            logger.warning("DAG→CI X/Y mapping ignored nodes not in dataset columns: %s", selected_ids)
            return no_update, no_update, cur_z
        logger.info("DAG→CI: set X=%s Y=%s from DAG selection", valid[0], valid[1])
        return valid[0], valid[1], cur_z

    if trigger_id == "dag-use-z":
        if not data_json_state:
            logger.warning("DAG→CI Z mapping skipped - no data available")
            return no_update, no_update, cur_z
        selected_ids = [n.get("id") for n in (selected_nodes or []) if n.get("id")]
        df = pd.read_json(StringIO(data_json_state), orient="split")
        df_cols = set(df.columns)
        valid = [n for n in selected_ids if n in df_cols]
        dropped = set(selected_ids) - set(valid)
        if dropped:
            logger.warning("DAG→CI Z mapping dropped nodes not in dataset columns: %s", list(dropped))
        logger.info("DAG→CI: set Z=%s from DAG selection", valid)
        return cur_x, cur_y, valid

    if trigger_id == "consistency-table" and consistency_data and selected_rows:
        row_index = selected_rows[0]
        if row_index < 0 or row_index >= len(consistency_data):
            return cur_x, cur_y, cur_z
        row = consistency_data[row_index]
        x_val = row.get("x")
        y_val = row.get("y")
        conds = row.get("conds") or []
        df = pd.read_json(StringIO(data_json_state), orient="split") if data_json_state else None
        df_cols = set(df.columns) if df is not None else set()
        valid_x = x_val if x_val in df_cols else None
        valid_y = y_val if y_val in df_cols else None
        valid_z = [c for c in conds if c in df_cols]
        if not valid_x or not valid_y:
            logger.warning("DAG vs Data row mapping skipped - nodes not in dataset columns")
            return cur_x, cur_y, cur_z
        logger.info("DAG vs Data row → CI: X=%s Y=%s Z=%s", valid_x, valid_y, valid_z)
        return valid_x, valid_y, valid_z

    return cur_x, cur_y, cur_z


def run_server(**kwargs):
    # Dash 3+ replaced run_server with run.
    app.run(**kwargs)


if __name__ == "__main__":
    run_server(debug=True)
