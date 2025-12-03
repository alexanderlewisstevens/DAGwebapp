import networkx as nx

from causal_playground.core import dag_model


def test_add_and_remove_nodes_and_edges():
    dag = dag_model.create_empty_dag()
    dag_model.add_node(dag, "A")
    dag_model.add_node(dag, "B")
    dag_model.add_edge(dag, "A", "B")
    assert dag.has_node("A") and dag.has_node("B")
    assert dag.has_edge("A", "B")

    dag_model.remove_edge(dag, "A", "B")
    assert not dag.has_edge("A", "B")
    dag_model.remove_node(dag, "A")
    assert not dag.has_node("A")
    assert dag.has_node("B")


def test_dag_to_figure_handles_simple_graph():
    dag = dag_model.create_empty_dag()
    dag_model.add_node(dag, "A")
    dag_model.add_node(dag, "B")
    dag_model.add_edge(dag, "A", "B")
    fig = dag_model.dag_to_figure(dag)
    assert fig is not None
    assert dag.number_of_nodes() == 2
    assert dag.number_of_edges() == 1


def test_cytoscape_conversion_round_trip():
    dag = dag_model.create_empty_dag()
    dag_model.add_node(dag, "A")
    dag_model.add_node(dag, "B")
    dag_model.add_edge(dag, "A", "B")
    elements = dag_model.dag_to_cytoscape_elements(dag)
    rebuilt = dag_model.cytoscape_elements_to_dag(elements)
    assert set(rebuilt.nodes) == {"A", "B"}
    assert rebuilt.has_edge("A", "B")


def test_d_separation_chain():
    dag = dag_model.create_empty_dag()
    dag_model.add_edge(dag, "A", "B")
    dag_model.add_edge(dag, "B", "C")
    assert not dag_model.is_d_separated(dag, "A", "C", [])
    assert dag_model.is_d_separated(dag, "A", "C", ["B"])


def test_d_separation_collider():
    dag = dag_model.create_empty_dag()
    dag_model.add_edge(dag, "A", "C")
    dag_model.add_edge(dag, "B", "C")
    assert dag_model.is_d_separated(dag, "A", "B", [])
    assert not dag_model.is_d_separated(dag, "A", "B", ["C"])


def test_implied_independencies_simple():
    dag = dag_model.create_empty_dag()
    dag_model.add_edge(dag, "A", "C")
    dag_model.add_edge(dag, "B", "C")
    independencies = dag_model.implied_independencies(dag, nodes=["A", "B", "C"], max_conditions=0)
    assert ("A", "B", ()) in independencies or ("B", "A", ()) in independencies


def test_dag_serialization_round_trip():
    dag = dag_model.create_empty_dag()
    dag_model.add_edge(dag, "A", "B")
    dag_model.add_edge(dag, "B", "C")
    payload = dag_model.dag_to_serializable(dag)
    rebuilt = dag_model.dag_from_serializable(payload)
    assert set(rebuilt.nodes) == {"A", "B", "C"}
    assert set(rebuilt.edges) == {("A", "B"), ("B", "C")}


def test_dag_from_columns():
    dag = dag_model.dag_from_columns(["A", "B", "C"])
    assert set(dag.nodes) == {"A", "B", "C"}
    assert dag.number_of_edges() == 0
