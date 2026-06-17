from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained COVID-19 X-ray model.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/best_model.keras"))
    parser.add_argument("--data-dir", type=Path, default=Path("../covid-19/DataSet/Data/test"))
    parser.add_argument("--class-map", type=Path, default=Path("artifacts/reports/class_mapping.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is required for evaluation. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model_path)
    ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    class_names = ds.class_names
    if args.class_map.exists():
        class_map = json.loads(args.class_map.read_text())
        class_names = [class_map[str(i)] for i in range(len(class_map))]

    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in ds]).astype(int)
    y_prob = model.predict(ds, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    payload = {
        "class_names": class_names,
        "threshold": 0.5,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    (args.output_dir / "evaluation_report.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
