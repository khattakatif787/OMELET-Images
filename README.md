# OMELET — Image Datasets

A framework for **safe image classification** using **ensembles of self-controlled components (SCCs)**.

It is designed for **image datasets** and supports the full experimental pipeline, from training deep classifiers to generating SCCs (where `-1` represents rejected outputs), analyzing SCC pair interactions, and building SCC ensembles.

This GitHub repository is built for image classification experiments using SCC-based safe ensemble methods.

---

## Features

- Train multiple deep image classifiers
- Compute uncertainty measures (`MaxProb`, `Entropy`, `Combined`, `Multi-Combined`, `Reconstruction Loss`)
- Generate SCC outputs under different ALR settings
- Compute performance metrics for classifiers and SCCs
- Analyze pairwise SCC gain/drop behavior
- Build SCC ensembles
- Build baseline SCC ensembles using SCCs sorted by highest individual accuracy, without gain/drop, for comparison

---

## Repository Structure

    OMELET-Images/
    ├── dataset/                             # Dataset resources
    ├── debug/
    │   ├── Med_dataset/                     # Saved outputs for one dataset / experiment
    │   ├── learn_dnn_misc_detector.py       # Train classifiers and compute uncertainty measures
    │   └── plmodels.py                      # Model definitions used during training
    ├── models/                              # Saved models
    ├── sprout/                              # Internal SPROUT package code
    ├── build_ensembles.py                   # Build SCC ensembles using the gain/drop concept
    ├── build_ensembles_sorted_accuracy.py   # Build baseline SCC ensembles using SCCs sorted by highest individual accuracy, without gain/drop, for comparison
    ├── compute_classifier_scc_stats.py      # Compute classifier and SCC performance metrics
    ├── generate_SCC_outputs.py              # Generate SCC outputs from classifier outputs
    ├── GenericDataset.py                    # Generic dataset loader
    ├── run_pipeline.sh                      # End-to-end pipeline script
    ├── run_scc_couples.py                   # Pairwise SCC analysis
    ├── LICENSE
    └── README.md
    

---

## Uncertainty Measures Used

The framework uses the following uncertainty measures for SCC generation:

- **Maximum Likelihood (`UncertaintyCalculator.MaxProbUncertainty`)**  
  Uses the maximum class probability. 

- **Entropy of Probabilities (`UncertaintyCalculator.EntropyUncertainty`)**  
  Computes the entropy of the predicted probability distribution. Higher entropy indicates higher uncertainty.

- **Combined Uncertainty (`UncertaintyCalculator.CombinedUncertainty`)**  
  Uses one checker classifier to validate the main classifier. Positive values indicate agreement, negative values indicate disagreement, and values near zero indicate maximum uncertainty.

- **Multi-Combined Uncertainty (`UncertaintyCalculator.MultiCombinedUncertainty`)**  
  Extends combined uncertainty by averaging the uncertainty contributions of multiple checker classifiers.

- **Reconstruction Loss (`UncertaintyCalculator.ReconstructionLoss`)**  
  Uses autoencoder reconstruction error to identify out-of-distribution or unseen samples. Higher values indicate higher uncertainty.

---

## Pipeline Overview

The framework assumes the dataset is split into:

- `train`
- `val`
- `test`

The pipeline works as follows:

1. **Train deep image classifiers**
   - Train multiple classifiers on the training set
   - Compute predictions, probabilities, and uncertainty measures on validation and test data
   - Store outputs in CSV format

2. **Generate SCC outputs**
   - For each classifier and uncertainty measure, determine the rejection threshold on the **validation** set
   - Apply the same threshold to the **test** set

3. **Build SCC ensembles**
   - Determine the SCC order using **validation** data
   - Apply the same SCC order to **test** data

4. **Analyze SCC interactions**
   - Evaluate pairwise SCC behavior using gain and drop metrics

5. **Compute performance metrics**
   - Evaluate classifiers, SCCs, SCC pairs, and ensembles using performance metrics

All outputs are finally organized under:

    debug/<DATASET>/

---

## Quick Start

Run the full pipeline with:

    bash run_pipeline.sh <DATASET>

### Example

    bash run_pipeline.sh tt100k

Before running, update the dataset path in:

    debug/learn_dnn_misc_detector.py

Set:

    TRAIN_DATA_FOLDER = "/path/to/your/dataset"

---

## Dataset Format

Your dataset should be organized as follows:

    root_dir/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    ├── val/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── test/
        ├── class_1/
        ├── class_2/
        └── ...

---

## Full Pipeline Steps

Running the full pipeline automatically performs:

1. Training deep image classifiers
2. Computing uncertainty measures
3. Generating SCC outputs using validation-selected thresholds
4. Building SCC ensembles using validation-selected SCC ordering
5. Building baseline SCC ensembles using SCCs sorted by individual accuracy
6. Computing classifier and SCC performance metrics
7. Running SCC pair analysis
8. Saving results under `debug/<DATASET>/`

---

## Manual Execution

You can also run the steps manually:

    python debug/learn_dnn_misc_detector.py
    python generate_SCC_outputs.py
    python build_ensembles.py --dataset CUSTOM --fcc_alrs 0.01 0.001 0.0001 --ensemble_sizes 2 3 4
    python build_ensembles_sorted_accuracy.py --dataset CUSTOM --fcc_alrs 0.01 0.001 0.0001 --ensemble_sizes 2 3 4
    python compute_classifier_scc_stats.py
    python run_scc_couples.py debug/SCC_outputs

`debug/SCC_outputs` is the folder where SCC outputs for both validation and test sets are stored.

---

## Evaluation Metrics

The repository uses the following metrics:

| Metric   | Description |
|----------|-------------|
| `aw`     | Accepted and correct predictions divided by the total number of samples |
| `phi`    | Rejection rate |
| `ew`     | Accepted but incorrect predictions divided by the total number of samples |
| `ew_ans` | Error rate among accepted predictions only |

Rejected predictions are represented by:

    -1

---

## Output Folders

During execution, the project may generate folders such as:

- `tmp/`
- `checkpoints/`
- `lightning_logs/`
- `trained_models/`
- `debug/SCC_outputs/`
- `debug/ENSEMBLE_outputs/`
- `debug/ENSEMBLE_outputs_accuracy_sorted/`
- `debug/SCC_couples/`

At the end of the pipeline, outputs are reorganized into:

    debug/<DATASET>/

This keeps experiment results separated by dataset.

---

## Supported Data

The current implementation is intended for **image classification experiments** and supports:

- benchmark datasets
- custom datasets following the required folder structure

---

## Dependencies

Minimum required packages include:

- Python 3.8+
- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `openpyxl`
- `Pillow`

The project also depends on internal modules such as:

- `plmodels`
- `SPROUTObject`
- `sprout.classifiers`
- `sprout.utils`
- `confens`

---

## Notes

- Threshold selection is performed on the **validation set**
- The selected thresholds are then applied to the **test** set
- SCC ordering for ensembles is determined on **validation data**
- The same order is reused on **test** data
- The accuracy-sorted ensemble script is included only as a baseline comparison against the gain/drop-based ensemble construction

This ensures a consistent validation-to-test workflow across the repository.

---

## License

This repository includes and adapts code from:

- SPROUT — MIT License
- OMELET — GPL-3.0 License

Original copyright and license notices are preserved.  
Since this repository includes GPL-3.0 licensed code, it is distributed under the GPL-3.0 license.