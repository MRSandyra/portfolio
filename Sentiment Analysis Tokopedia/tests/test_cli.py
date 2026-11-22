import json

from tokopedia_sentiment.cli import main


def test_demo_runs_without_external_dataset(capsys) -> None:
    exit_code = main(["demo", "produk bagus", "barang rusak"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(payload["predictions"]) == 2
    assert "not quality evidence" in payload["warning"]
