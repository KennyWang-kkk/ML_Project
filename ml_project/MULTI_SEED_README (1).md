# Multi-Seed Experiment (`multi_seed.ipynb`)

This notebook extends the original `adversarial_kd_mnist.py` by running the full experiment across **3 independent random seeds** to produce statistically reliable results.

## What's New

- **Multi-seed training** — all 5 models (teacher, baseline, standard_kd, adv_training, adv_kd) are trained 3 times with seeds 42, 43, 44
- **Repeated adversarial evaluation** — each model's FGSM and PGD evaluation is repeated 3 times per run and averaged, reducing noise from PGD's random initialization
- **Summary statistics** — results are aggregated into `summary_mean_std.csv` (mean ± std across training seeds)
- **Colab-ready** — automatically clones the repo from GitHub and installs dependencies; designed to run on Google Colab A100

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Epochs per run | 15 |
| Batch size | 128 |
| Seeds | 42, 43, 44 |
| PGD steps | 40 |
| PGD step size | 0.01 |
| Eval repeats | 3 |
| Training runs | 3 |

## Output Structure

```
outputs/multi_seed_repeated_YYYYMMDD_HHMMSS/
├── multi_seed_config.json       # full experiment config
├── summary_mean_std.csv         # mean ± std across all 3 seeds  ← main result
├── run_1_seed_42/
│   ├── attack_results.csv
│   ├── fgsm_accuracy_vs_epsilon.png
│   ├── pgd_accuracy_vs_epsilon.png
│   ├── training_loss_curves.png
│   └── *.pt                     # saved model weights
├── run_2_seed_43/
└── run_3_seed_44/
```

Estimated runtime on A100: **~30 minutes**
