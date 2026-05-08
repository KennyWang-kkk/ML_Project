# Multi-Seed MNIST Robustness Experiment

This folder contains the multi-seed version of the MNIST adversarial robustness experiment. It extends `src/adversarial_kd_mnist.py` by running the full training and evaluation pipeline across multiple random seeds, then aggregating the results into mean and standard deviation statistics.

## Purpose

The goal of this experiment is to make the reported results more reliable than a single-seed run. Instead of reporting one training result, the notebook trains and evaluates each method under several independent random seeds and summarizes the average performance.

The experiment compares the following models:

- Teacher model
- Baseline model
- Standard knowledge distillation model
- Adversarial training model
- Adversarial knowledge distillation model

Each model is evaluated on clean accuracy, FGSM robustness, and PGD robustness.

## Files

Recommended location in the final project repository:

```text
experiments/multi_seed/
├── README.md
└── multi_seed.ipynb
```

This notebook expects the project repository to contain:

```text
src/
└── adversarial_kd_mnist.py
```

The notebook imports the main model, attack, training, and evaluation functions from this source file. Therefore, `src/adversarial_kd_mnist.py` must be included in the final repository.

## Experiment Configuration

| Parameter | Value |
|---|---:|
| Epochs per run | 15 |
| Batch size | 128 |
| Random seeds | 42, 43, 44 |
| PGD steps | 40 |
| PGD step size | 0.01 |
| Evaluation repeats | 3 |
| Total training runs | 3 |

## Output Structure

Running the notebook creates an output folder similar to:

```text
outputs/multi_seed_repeated_YYYYMMDD_HHMMSS/
├── multi_seed_config.json
├── summary_mean_std.csv
├── run_1_seed_42/
│   ├── attack_results.csv
│   ├── fgsm_accuracy_vs_epsilon.png
│   ├── pgd_accuracy_vs_epsilon.png
│   ├── training_loss_curves.png
│   └── *.pt
├── run_2_seed_43/
└── run_3_seed_44/
```

The most important result file is:

```text
summary_mean_std.csv
```

This file reports the mean and standard deviation across the three random seeds.

## Recommended Files to Commit

For the final GitHub submission, commit the notebook, README, result tables, and result figures:

```text
experiments/multi_seed/multi_seed.ipynb
experiments/multi_seed/README.md
results/multi_seed/summary_mean_std.csv
results/multi_seed/fgsm_accuracy_vs_epsilon.png
results/multi_seed/pgd_accuracy_vs_epsilon.png
results/multi_seed/training_loss_curves.png
```

Large model checkpoints such as `.pt` files should usually not be committed unless the instructor explicitly requires them. They can make the repository unnecessarily large.

A suitable `.gitignore` entry is:

```text
outputs/
*.pt
*.pth
__pycache__/
.ipynb_checkpoints/
```

## Running the Notebook

From the project root, open:

```text
experiments/multi_seed/multi_seed.ipynb
```

Then run all cells. The notebook should use the current repository code rather than cloning another copy of the project.

If running in Google Colab, upload or mount the final project repository first, then run the notebook from inside that repository.

## Notes

- The experiment is computationally heavier than the single-seed version because it trains all models three times.
- PGD evaluation is repeated several times to reduce noise from random initialization.
- The expected runtime depends on GPU type. On an A100-class GPU, the full experiment is expected to finish in roughly tens of minutes.
