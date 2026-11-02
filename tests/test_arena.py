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
