from vajra.cli import evaluate_synthetic, main


def test_radar_extrapolation_warns_earlier_than_lightning_only_for_new_storms():
    rows, gained, n_new = evaluate_synthetic(seed=0, leads=(30,))
    assert {r[0] for r in rows} == {"Persistence", "Lightning-only tracking", "Radar extrapolation"}
    assert n_new > 0 and gained > 0


def test_cap_example_prints_an_exercise_alert(capsys):
    assert main(["cap-example"]) == 0
    out = capsys.readouterr().out
    assert "<alert" in out and "Exercise" in out
