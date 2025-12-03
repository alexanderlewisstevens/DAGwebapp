import pandas as pd

from causal_playground.core import ci_engine, dag_model


def test_smoke_dag_and_ci_flow():
    # Create a tiny dataset resembling A->B->C with D noise.
    df = pd.DataFrame(
        {
            "A": [0, 0, 1, 1],
            "B": [0, 1, 1, 1],
            "C": [0, 1, 1, 1],
            "D": [0, 1, 0, 1],
        }
    )

    # Auto-build DAG from columns
    dag = dag_model.dag_from_columns(list(df.columns))
    assert set(dag.nodes) == {"A", "B", "C", "D"}
    assert dag.number_of_edges() == 0

    # Add edges matching the story
    dag_model.add_edge(dag, "A", "B")
    dag_model.add_edge(dag, "B", "C")

    # Implied independencies should include A _|_ C | B for this chain
    implied = dag_model.implied_independencies(dag, nodes=["A", "B", "C"], max_conditions=1)
    assert ("A", "C", ("B",)) in implied or ("C", "A", ("B",)) in implied

    # CI test should run without error
    results = ci_engine.test_independencies(df, [("A", "C", ("B",))], alpha=0.5)
    assert not results.empty
    assert set(results.columns) >= {"x", "y", "conds", "p_value", "decision", "n"}
