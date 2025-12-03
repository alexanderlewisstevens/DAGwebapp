import pandas as pd

from causal_playground.core.ci_engine import conditional_ci_summary, contingency_details


def test_conditional_ci_summary_basic():
    df = pd.DataFrame(
        {
            "x": ["a", "a", "b", "b", "a", "b"],
            "y": ["u", "v", "u", "v", "u", "u"],
            "z": ["c1", "c1", "c1", "c2", "c2", "c2"],
        }
    )
    summary = conditional_ci_summary(df, "x", "y", ["z"])
    assert len(summary) == 2
    assert set(summary["condition"]) == {"z=c1", "z=c2"}
    assert summary["p"].between(0, 1).all()
    assert (summary["n"] > 0).all()


def test_conditional_ci_summary_no_conditions():
    df = pd.DataFrame({"x": ["a", "a", "b"], "y": ["u", "v", "u"]})
    summary = conditional_ci_summary(df, "x", "y", [])
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["condition"] == "All"
    assert row["n"] == 3
    assert 0 <= row["p"] <= 1


def test_conditional_ci_summary_multiple_conditions():
    df = pd.DataFrame(
        {
            "x": ["a", "a", "b", "b"],
            "y": ["u", "v", "u", "v"],
            "z1": ["c1", "c1", "c2", "c2"],
            "z2": ["d1", "d2", "d1", "d2"],
        }
    )
    summary = conditional_ci_summary(df, "x", "y", ["z1", "z2"])
    assert len(summary) == 4
    assert summary["condition"].str.contains("z1=").all()
    assert summary["condition"].str.contains("z2=").all()
    assert summary["p"].between(0, 1).all()


def test_contingency_details_probabilities_sum_to_one():
    df = pd.DataFrame({"x": ["a", "a", "b"], "y": ["u", "v", "u"]})
    details = contingency_details(df, "x", "y")
    assert abs(details["joint_probs"].values.sum() - 1.0) < 1e-9


def test_test_independencies():
    df = pd.DataFrame(
        {
            "x": ["a", "a", "b", "b"],
            "y": ["u", "v", "u", "v"],
            "z": ["c1", "c1", "c2", "c2"],
        }
    )
    independencies = [("x", "y", ()), ("x", "y", ("z",))]
    from causal_playground.core.ci_engine import test_independencies

    tested = test_independencies(df, independencies, alpha=0.05)
    assert not tested.empty
    assert set(tested.columns) >= {"x", "y", "conds", "p_value", "decision", "n"}


def test_test_independencies_respects_alpha():
    df = pd.DataFrame({"x": ["a", "a", "b", "b"], "y": ["u", "v", "u", "v"]})
    independencies = [("x", "y", ())]
    from causal_playground.core.ci_engine import test_independencies

    tested = test_independencies(df, independencies, alpha=0.5)
    assert (tested["decision"] == "fail_to_reject").all()


def test_contingency_probs_returned_in_summary():
    df = pd.DataFrame({"x": ["a", "a", "b"], "y": ["u", "v", "u"]})
    summary = conditional_ci_summary(df, "x", "y", [])
    assert "contingency_probs" in summary.columns
    probs = summary.iloc[0]["contingency_probs"]
    assert abs(sum(sum(row) for row in probs) - 1.0) < 1e-6
