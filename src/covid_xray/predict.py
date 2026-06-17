from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict COVID-19 vs normal from one X-ray image.")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/best_model.keras"))
    parser.add_argument("--class-map", type=Path, default=Path("artifacts/reports/class_mapping.json"))
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is required for prediction. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    model = tf.keras.models.load_model(args.model_path)

    class_names = ["COVID19", "NORMAL"]
    if args.class_map.exists():
        class_map = json.loads(args.class_map.read_text())
        class_names = [class_map[str(i)] for i in range(len(class_map))]

    image = tf.keras.utils.load_img(args.image_path, target_size=tuple(args.image_size))
    array = tf.keras.utils.img_to_array(image)
    batch = np.expand_dims(array, axis=0)

    probability_class_1 = float(model.predict(batch, verbose=0)[0][0])
    predicted_index = int(probability_class_1 >= args.threshold)
    result = {
        "image": str(args.image_path),
        "predicted_class": class_names[predicted_index],
        "probability_class_1": probability_class_1,
        "class_1": class_names[1],
        "threshold": args.threshold,
        "note": "Research prototype only; not for clinical diagnosis.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
