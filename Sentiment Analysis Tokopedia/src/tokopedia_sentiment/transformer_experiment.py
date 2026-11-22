"""Optional IndoBERT-Lite experiment; heavy dependencies are imported lazily."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tokopedia_sentiment.case_study_data import (
    CLASS_NAMES,
    prepare_case_study_reviews,
    temporal_split,
)
from tokopedia_sentiment.case_study_evaluation import evaluate_multiclass
from tokopedia_sentiment.decision import predict_with_thresholds, select_minority_thresholds
from tokopedia_sentiment.triage import apply_triage_policy, evaluate_triage, select_triage_policy


@dataclass(frozen=True)
class TransformerConfig:
    data_path: Path
    output_directory: Path
    report_path: Path
    model_name: str = "indobenchmark/indobert-lite-base-p1"
    epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_length: int = 128
    positive_to_minority_ratio: float = 5.0
    seed: int = 42
    cache_directory: Path = Path(".model-cache/huggingface")


def stratified_transformer_training_sample(
    frame: pd.DataFrame, *, positive_to_minority_ratio: float, seed: int
) -> pd.DataFrame:
    """Keep every rare row and cap only the positive training majority."""
    minority = frame.loc[frame["label"] != 2]
    positive = frame.loc[frame["label"] == 2]
    positive_count = min(len(positive), max(1, round(len(minority) * positive_to_minority_ratio)))
    sampled_positive = positive.sample(n=positive_count, random_state=seed)
    return (
        pd.concat([minority, sampled_positive], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def _predict_probabilities(model: Any, loader: Any, device: Any, torch: Any) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels", None)
            del labels
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.concatenate(outputs)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def run_transformer_experiment(config: TransformerConfig) -> dict[str, Any]:
    """Fine-tune IndoBERT-Lite without adding its model/cache to version control."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, BertTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install the optional transformer dependencies with `uv sync --extra bert`."
        ) from error

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = pd.read_csv(config.data_path)
    prepared, preparation = prepare_case_study_reviews(raw)
    splits = temporal_split(prepared)
    training = stratified_transformer_training_sample(
        splits.train,
        positive_to_minority_ratio=config.positive_to_minority_ratio,
        seed=config.seed,
    )
    config.cache_directory.mkdir(parents=True, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_directory)

    class ReviewDataset(Dataset):
        def __init__(self, frame: pd.DataFrame) -> None:
            self.text = frame["text"].tolist()
            self.labels = frame["label"].tolist()

        def __len__(self) -> int:
            return len(self.text)

        def __getitem__(self, index: int) -> tuple[str, int]:
            return self.text[index], self.labels[index]

    def collate(batch: list[tuple[str, int]]) -> dict[str, Any]:
        text, labels = zip(*batch, strict=True)
        encoded = tokenizer(
            list(text),
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        ReviewDataset(training),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        ReviewDataset(splits.validation),
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        ReviewDataset(splits.test),
        batch_size=config.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        cache_dir=config.cache_directory,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    counts = np.bincount(training["label"].to_numpy(), minlength=3)
    class_weights = len(training) / (3 * counts)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_state: dict[str, Any] | None = None
    best_macro_f1 = -1.0
    history = []
    validation_actual = splits.validation["label"].to_numpy()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
        validation_probability = _predict_probabilities(model, validation_loader, device, torch)
        validation_predicted = np.argmax(validation_probability, axis=1)
        from sklearn.metrics import f1_score

        macro_f1 = float(f1_score(validation_actual, validation_predicted, average="macro"))
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": total_loss / len(train_loader),
                "validation_macro_f1_argmax": macro_f1,
            }
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("transformer training did not complete an epoch")
    model.load_state_dict(best_state)
    model.to(device)

    validation_probability = _predict_probabilities(model, validation_loader, device, torch)
    thresholds = select_minority_thresholds(
        validation_actual, validation_probability, CLASS_NAMES, target_recall=0.70
    )
    triage_policy = select_triage_policy(validation_actual, validation_probability, CLASS_NAMES)
    test_probability = _predict_probabilities(model, test_loader, device, torch)
    test_actual = splits.test["label"].to_numpy()
    test_predicted = predict_with_thresholds(
        test_probability,
        CLASS_NAMES,
        thresholds.thresholds,
        thresholds.majority_class,
    )
    triage_decisions = apply_triage_policy(test_probability, CLASS_NAMES, triage_policy)
    report = {
        "config": {
            **asdict(config),
            "data_path": str(config.data_path),
            "output_directory": str(config.output_directory),
            "report_path": str(config.report_path),
            "cache_directory": str(config.cache_directory),
        },
        "device": str(device),
        "preparation": preparation.to_dict(),
        "training_rows": len(training),
        "training_class_distribution": training["label_name"].value_counts().to_dict(),
        "history": history,
        "decision_policy": thresholds.to_dict(),
        "triage_policy": triage_policy.to_dict(),
        "test_metrics": evaluate_multiclass(
            test_actual, test_probability, test_predicted, CLASS_NAMES
        ),
        "triage_metrics": evaluate_triage(test_actual, triage_decisions, CLASS_NAMES),
        "warning": "All labels are rating-derived proxies; this is not human-validated.",
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_directory, safe_serialization=True)
    tokenizer.save_pretrained(config.output_directory)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
