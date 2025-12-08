"""Minimal DAG helpers built on top of networkx."""

from __future__ import annotations

import logging
import matplotlib

# Use a non-interactive backend to avoid GUI requirements in server contexts.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx
from pgmpy.base import DAG as PgmpyDAG

logger = logging.getLogger("causal_playground")


def create_empty_dag() -> nx.DiGraph:
    """Create an empty directed acyclic graph container."""
    dag = nx.DiGraph()
    logger.info("Created empty DAG")
    return dag


def add_node(dag: nx.DiGraph, node_name: str) -> nx.DiGraph:
    dag.add_node(node_name)
    logger.info("Added node '%s' - nodes=%s", node_name, dag.number_of_nodes())
    return dag


def remove_node(dag: nx.DiGraph, node_name: str) -> nx.DiGraph:
    dag.remove_node(node_name)
    logger.info("Removed node '%s' - nodes=%s", node_name, dag.number_of_nodes())
    return dag


def add_edge(dag: nx.DiGraph, src: str, dst: str) -> nx.DiGraph:
    dag.add_edge(src, dst)
    logger.info("Added edge %s -> %s - edges=%s", src, dst, dag.number_of_edges())
    return dag


def remove_edge(dag: nx.DiGraph, src: str, dst: str) -> nx.DiGraph:
    dag.remove_edge(src, dst)
    logger.info("Removed edge %s -> %s - edges=%s", src, dst, dag.number_of_edges())
    return dag


def dag_to_figure(dag: nx.DiGraph):
    """Render the DAG using networkx + matplotlib for Phase 1."""
    logger.info("Rendering DAG - nodes=%s edges=%s", dag.number_of_nodes(), dag.number_of_edges())
    fig, ax = plt.subplots(figsize=(5, 4))
    if dag.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "DAG is empty", ha="center", va="center")
        ax.axis("off")
        return fig
    pos = nx.spring_layout(dag, seed=42)
    nx.draw_networkx(
        dag,
        pos=pos,
        ax=ax,
        arrows=True,
        node_color="#90caf9",
        edge_color="#424242",
        font_size=10,
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def dag_to_cytoscape_elements(dag: nx.DiGraph) -> list[dict]:
    """Convert a networkx DAG to dash-cytoscape elements."""
    nodes = [{"data": {"id": node, "label": node}} for node in dag.nodes]
    edges = [{"data": {"source": u, "target": v}} for u, v in dag.edges]
    logger.info(
        "Converted DAG to cytoscape elements - nodes=%s edges=%s",
        len(nodes),
        len(edges),
    )
    return nodes + edges


def cytoscape_elements_to_dag(elements: list[dict]) -> nx.DiGraph:
    """Reconstruct a DAG from dash-cytoscape elements."""
    dag = create_empty_dag()
    for el in elements or []:
        data = el.get("data", {})
        if "source" in data and "target" in data:
            add_edge(dag, data["source"], data["target"])
        elif "id" in data:
            add_node(dag, data["id"])
    logger.info(
        "Converted cytoscape elements to DAG - nodes=%s edges=%s",
        dag.number_of_nodes(),
        dag.number_of_edges(),
    )
    return dag


def dag_to_serializable(dag: nx.DiGraph) -> dict:
    """Serialize a DAG to a JSON-friendly dict."""
    return {"nodes": list(dag.nodes), "edges": [[u, v] for u, v in dag.edges]}


def dag_from_serializable(payload: dict) -> nx.DiGraph:
    """Construct a DAG from a serialized dict with nodes/edges."""
    dag = create_empty_dag()
    for node in payload.get("nodes", []):
        add_node(dag, node)
    for edge in payload.get("edges", []):
        if len(edge) == 2:
            add_edge(dag, edge[0], edge[1])
    return dag


def dag_from_columns(columns: list[str]) -> nx.DiGraph:
    """Create a DAG with one node per column name, no edges."""
    dag = create_empty_dag()
    for col in columns:
        add_node(dag, col)
    logger.info("Initialized DAG from columns: nodes=%s edges=%s", dag.number_of_nodes(), dag.number_of_edges())
    return dag


def _to_pgmpy_dag(dag: nx.DiGraph) -> PgmpyDAG:
    """Convert networkx DiGraph to pgmpy DAG."""
    pg_dag = PgmpyDAG()
    pg_dag.add_nodes_from(dag.nodes)
    pg_dag.add_edges_from(dag.edges)
    return pg_dag


def is_d_separated(dag: nx.DiGraph, x: str, y: str, conds: list[str]) -> bool:
    """Return True if x and y are d-separated given conds in this DAG."""
    pg_dag = _to_pgmpy_dag(dag)
    separated = not pg_dag.is_dconnected(x, y, observed=conds or None)
    logger.debug("D-separation check - x=%s y=%s conds=%s separated=%s", x, y, conds, separated)
    return separated


def implied_independencies(
    dag: nx.DiGraph, nodes: list[str] | None = None, max_conditions: int = 1
) -> list[tuple[str, str, tuple[str, ...]]]:
    """
    Enumerate graph-implied independencies via d-separation.

    - nodes: optional subset of nodes to consider.
    - max_conditions: include conditioning sets up to this size (default 1).
    """
    consider_nodes = list(nodes) if nodes is not None else list(dag.nodes)
    independencies: list[tuple[str, str, tuple[str, ...]]] = []
    for i, x in enumerate(consider_nodes):
        for y in consider_nodes[i + 1 :]:
            other_nodes = [n for n in consider_nodes if n not in {x, y}]
            # unconditional
            if is_d_separated(dag, x, y, []):
                independencies.append((x, y, ()))
            if max_conditions >= 1:
                for z in other_nodes:
                    if is_d_separated(dag, x, y, [z]):
                        independencies.append((x, y, (z,)))
    logger.info(
        "Implied independencies generated - count=%s nodes_considered=%s max_conditions=%s",
        len(independencies),
        len(consider_nodes),
        max_conditions,
    )
    return independencies
