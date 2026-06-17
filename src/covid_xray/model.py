from __future__ import annotations

import tensorflow as tf


def build_model(
    image_size: tuple[int, int],
    learning_rate: float,
    use_imagenet: bool = False,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    Build a binary X-ray classifier.

    The model uses MobileNetV2 as a compact convolutional backbone. ImageNet
    weights are optional because this project should also run offline.
    """
    inputs = tf.keras.Input(shape=(*image_size, 3), name="xray_image")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )
    x = augmentation(inputs)

    if use_imagenet:
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    else:
        x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    weights = "imagenet" if use_imagenet else None
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights=weights,
    )
    # Only freeze when using meaningful pretrained weights. If weights are
    # random, the backbone must train or the classifier head learns from noise.
    base.trainable = (not freeze_base) or (not use_imagenet)

    x = base(x, training=base.trainable)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="normal_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="covid_xray_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
