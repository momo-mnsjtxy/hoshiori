"""成对擂台：Elo 方向、平局折半，以及 Bradley-Terry 与 Elo 的排序一致性。"""

from features import arena, suite


def test_elo_moves_toward_repeated_winner():
    first, second = arena.elo([1.0] * 20)
    assert first > arena.ELO_INITIAL > second
    assert first + second == arena.ELO_INITIAL * 2


def test_ties_keep_equal_strength():
    outcomes = [0.5] * 12
    assert arena.elo(outcomes) == (1500.0, 1500.0)
    assert arena.bradley_terry(outcomes) == (0.5, 0.5)


def test_bradley_terry_order_agrees_with_elo():
    ranked = arena.rank(("strong {prompt}", "weak {prompt}"),
                        [1.0, 1.0, 0.5, 1.0, 0.0, 1.0])
    assert [row["backend"] for row in ranked] == ["strong {prompt}", "weak {prompt}"]
    assert ranked[0]["elo"] > ranked[1]["elo"]
    assert ranked[0]["bt"] > ranked[1]["bt"]


def test_run_randomizes_call_order_but_keeps_a_b_scores(monkeypatch, tmp_path):
    tasks = [suite.Task(id=f"q{number}", prompt=str(number), rule="exact", answer="A")
             for number in range(8)]
    one = suite.Suite(name="demo", desc="", tasks=tasks, path=tmp_path / "demo.yaml")

    def answer(command, prompt):
        return "A" if command == "first" else "B"

    monkeypatch.setattr(arena.lm_eval, "answer", answer)
    result = arena.run("first", "second", [one], seed=0)
    assert {tuple(match["order"]) for match in result["matches"]} == {("A", "B"), ("B", "A")}
    assert all(match["players"][0]["score"] == 1.0 for match in result["matches"])
    assert result["ranking"][0]["backend"] == "first"


def test_configurable_elo_and_stability_stop(monkeypatch, tmp_path):
    """平局使窗口稳定后停止，实际后端调用次数与已记录场数一致。"""
    calls = []  # 记录真正执行多少次答题
    monkeypatch.setattr(arena.lm_eval, "answer", lambda command, prompt: calls.append(command) or "same")
    tasks = [suite.Task(id=str(index), prompt="q", rule="exact", answer="same") for index in range(100)]
    one = suite.Suite(name="stable", desc="", tasks=tasks, path=tmp_path / "stable.yaml")
    result = arena.run("a", "b", [one], k=12, initial=1000, min_matches=8, window=4, tolerance=0.01)
    assert result["stopping"]["converged"] is True
    assert len(result["matches"]) == 8 and len(calls) == 16
    assert result["elo"] == {"k": 12, "initial": 1000}
    assert [row["elo"] for row in result["ranking"]] == [1000, 1000]
    assert arena.elo([1], k=20, initial=800) == (810, 790)
