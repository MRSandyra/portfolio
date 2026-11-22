import json

import pandas as pd

from tokopedia_sentiment.modeling import load_artifact
from tokopedia_sentiment.training import TrainingConfig, train


def test_training_writes_provenance_artifact_and_evaluation_reports(tmp_path) -> None:
    rows = []
    for index in range(20):
        rows.append({"Review": f"produk buruk kecewa kode {index}", "Rating": 1})
        rows.append({"Review": f"produk bagus puas nomor {index}", "Rating": 5})
    data_path = tmp_path / "reviews.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    artifact_path = tmp_path / "artifacts" / "model.joblib"
    report_directory = tmp_path / "reports"

    metadata = train(
        TrainingConfig(
            data_path=data_path,
            artifact_path=artifact_path,
            report_directory=report_directory,
            source_uri="synthetic://pytest",
        )
    )

    assert metadata["seed"] == 42
    assert metadata["source_uri"] == "synthetic://pytest"
    assert len(metadata["source_sha256"]) == 64
    assert metadata["split"]["rows"] == {"train": 28, "validation": 6, "test": 6}
    assert set(metadata["test_metrics"]["per_class"]) == {"negative", "positive"}
    assert load_artifact(artifact_path)["metadata"]["source_sha256"] == metadata["source_sha256"]
    assert json.loads((report_directory / "metrics.json").read_text(encoding="utf-8"))[
        "confusion_matrix"
    ]
    assert (report_directory / "confusion_matrix.png").is_file()
    assert (report_directory / "calibration_curve.png").is_file()
