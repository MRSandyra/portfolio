import json
from pathlib import Path


def test_notebook_is_a_thin_reproducible_walkthrough() -> None:
    notebook = json.loads(Path("Global Income Inequality.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert "/content/" not in source
    assert "def calculate_gini" not in source
    assert "from global_inequality" in source
    assert "not policy forecasts" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

