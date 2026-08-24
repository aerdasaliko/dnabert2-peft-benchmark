This repository provides the implementation used for benchmarking **parameter-efficient fine-tuning (PEFT)** methods for **DNABERT-2** on genomic prediction tasks.

The codebase is organized as a set of modular, composable scripts rather than a single unified pipeline. Scripts are used in combination depending on the dataset and experimental configuration.

---

## Data Preparation

* **Classification**: Datasets are publicly available and must be downloaded separately
[http://www.csbio.sjtu.edu.cn/bioinf/RBPsuite/dataset_new.html#start](http://www.csbio.sjtu.edu.cn/bioinf/RBPsuite/dataset_new.html#start)
* **Regression**: datasets are constructed using the provided `get-regression-dataset*.py` scripts

Preprocessing is performed using scripts in `dataset-preprocessing/`, which are composed as needed per dataset and task. No fixed pipeline is assumed, instead, scripts are combined depending on the experimental setting.

---

## Model Fine-Tuning

All experiments are conducted using **DNABERT-2** with PEFT-based adaptation.

Training is implemented in:

* `train-peft-classification.py` (classification setup)
* `train-peft-regression.py` (regression setup)

Benchmarking runs are executed via `finetune/scripts/run_peft_benchmark.sh`, which launches experiments across PEFT configurations and logs training metrics. Additional scripts support result merging and post-processing where required.

---

## Evaluation

Performance metrics are aggregated using scripts in `summarize-stats/`, which produce:

* summary tables across methods and datasets
* regression and classification metric reports
* prediction–ground-truth visualizations

---

## Reproducibility Notes

* The repository consists of **interoperable scripts rather than a fixed pipeline**.
* All scripts rely on **explicit global path variables**, which must be adjusted to the local environment prior to execution.
* Experiments assume a DNABERT-2-compatible runtime with GPU support and metric logging enabled during training.

