from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(
    train_dir: Path,
    test_dir: Path,
    image_size: tuple[int, int],
    batch_size: int,
    validation_split: float,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build stratified train, validation, and test datasets from class folders."""
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    train_paths, train_labels = _collect_paths_and_labels(train_dir, class_names)
    test_paths, test_labels = _collect_paths_and_labels(test_dir, class_names)

    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=validation_split,
        random_state=seed,
        stratify=train_labels,
    )

    train_ds = _make_dataset(
        tr_paths,
        tr_labels,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = _make_dataset(
        val_paths,
        val_labels,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_ds = _make_dataset(
        test_paths,
        test_labels,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    print(f"Found {len(train_paths)} training-directory files belonging to {len(class_names)} classes.")
    print(f"Using {len(tr_paths)} files for training.")
    print(f"Using {len(val_paths)} files for validation.")
    print(f"Found {len(test_paths)} test files belonging to {len(class_names)} classes.")
    return train_ds, val_ds, test_ds, class_names


def _collect_paths_and_labels(root: Path, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    paths: list[str] = []
    labels: list[int] = []
    for idx, class_name in enumerate(class_names):
        for path in sorted((root / class_name).iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(path))
                labels.append(idx)
    return np.asarray(paths), np.asarray(labels, dtype=np.float32)


def _make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_image(p, image_size), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def _load_image(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    return tf.cast(image, tf.float32)


def compute_class_weight(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Return balanced Keras class weights from folder counts."""
    counts = []
    for name in class_names:
        counts.append(sum(1 for p in (train_dir / name).iterdir() if p.is_file()))

    counts_arr = np.asarray(counts, dtype=np.float32)
    total = float(counts_arr.sum())
    n_classes = float(len(class_names))
    return {i: total / (n_classes * count) for i, count in enumerate(counts_arr)}


def save_class_mapping(class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(i): name for i, name in enumerate(class_names)}
    output_path.write_text(json.dumps(payload, indent=2))
