# COVID-19 Chest X-ray Classifier

Binary image classification project for classifying chest X-rays as `COVID19`
or `NORMAL`. This is a research/learning prototype and must not be used for
clinical diagnosis.

## Project Structure

```text
../covid-19/DataSet/Data/
  train/
    COVID19/
    NORMAL/
  test/
    COVID19/
    NORMAL/
src/covid_xray/
  train.py       # reproducible model training
  evaluate.py    # test-set report and confusion matrix
  predict.py     # single-image inference
  data.py        # dataset loading and class weights
  model.py       # model architecture
artifacts/
  models/
  reports/
  logs/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs this project into the active virtual environment
in editable mode. After that, Python can find the `covid_xray` package without
setting `PYTHONPATH=src`.

## Train

Offline-friendly training with randomly initialized MobileNetV2 backbone:

```bash
python -m covid_xray.train --epochs 20
```

Use ImageNet transfer learning when weights are available:

```bash
python -m covid_xray.train --epochs 20 --imagenet
```

Outputs:

```text
artifacts/models/best_model.keras
artifacts/models/final_model.keras
artifacts/reports/class_mapping.json
artifacts/reports/training_summary.json
artifacts/logs/training_log.csv
```

## Evaluate

```bash
python -m covid_xray.evaluate --model-path artifacts/models/best_model.keras
```

Output:

```text
artifacts/reports/evaluation_report.json
```

## Predict One Image

```bash
python -m covid_xray.predict \
  "../covid-19/DataSet/Data/test/COVID19/COVID19(8).jpg" \
  --model-path artifacts/models/best_model.keras
```

## Results

Evaluated on the held-out test set of 484 chest X-rays (167 COVID-19, 317 normal) using `best_model.keras` at a 0.5 threshold.

| Metric | Value |
| --- | ---: |
| Test accuracy | 98.35% |
| ROC-AUC | 0.996 |
| Weighted F1 | 0.984 |
| COVID-19 recall | 99.4% (166 of 167) |
| COVID-19 precision | 96.0% |
| Normal recall | 97.8% (310 of 317) |

**Confusion matrix (rows = actual, columns = predicted)**

|  | Predicted COVID-19 | Predicted Normal |
| --- | ---: | ---: |
| **Actual COVID-19** | 166 | 1 |
| **Actual Normal** | 7 | 310 |

Full metrics are saved in `artifacts/reports/evaluation_report.json` and `artifacts/reports/training_summary.json`.

> Research and learning prototype only. Not for clinical diagnosis.

## Original Notebook Result (earlier custom CNN)

The original notebook trained a custom CNN for 35 epochs and reported:

```text
Test loss: 0.0897
Test accuracy: 96.90%
```

The new scripts are intended to make that workflow reproducible and easier to
extend, evaluate, and explain.
