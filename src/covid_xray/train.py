from __future__ import annotations

import argparse
import json
from pathlib import Path

from covid_xray.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    cfg = ProjectConfig()
    parser = argparse.ArgumentParser(description="Train the COVID-19 X-ray classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=cfg.data_dir,
        help="Dataset root with train/ and test/ folders. Defaults to DataSet/Data.",
    )
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--image-size", type=int, nargs=2, default=cfg.image_size)
    parser.add_argument("--validation-split", type=float, default=cfg.validation_split)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--imagenet", action="store_true", help="Use ImageNet backbone weights.")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Fine-tune the full backbone.")
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="Use balanced class weights. Off by default because it can overcorrect small datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf
        from covid_xray.data import build_datasets, compute_class_weight, save_class_mapping
        from covid_xray.model import build_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is required for training. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    tf.keras.utils.set_random_seed(args.seed)

    output_dir = args.output_dir
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    logs_dir = output_dir / "logs"
    for directory in (models_dir, reports_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    image_size = tuple(args.image_size)

    train_ds, val_ds, test_ds, class_names = build_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )
    save_class_mapping(class_names, reports_dir / "class_mapping.json")
    class_weight = compute_class_weight(train_dir, class_names) if args.class_weight else None

    model = build_model(
        image_size=image_size,
        learning_rate=args.learning_rate,
        use_imagenet=args.imagenet,
        freeze_base=not args.unfreeze_backbone,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=models_dir / "best_model.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(logs_dir / "training_log.csv"),
        tf.keras.callbacks.TensorBoard(log_dir=str(logs_dir / "tensorboard")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save(models_dir / "final_model.keras")
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)

    report = {
        "class_names": class_names,
        "image_size": list(image_size),
        "batch_size": args.batch_size,
        "epochs_run": len(history.history["loss"]),
        "class_weight": None
        if class_weight is None
        else {str(k): float(v) for k, v in class_weight.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    (reports_dir / "training_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
