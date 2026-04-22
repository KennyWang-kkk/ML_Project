# CS 475/675 Final Project Framework

Project idea:

**Adversarial Knowledge Distillation for MNIST robustness**

The goal is to compare how different training methods behave under adversarial attacks. We train ordinary CNN models on MNIST, attack them with FGSM and PGD, and then test whether knowledge distillation plus adversarial examples improves robustness.

## What This Framework Includes

- Baseline CNN training on MNIST
- FGSM attack
- PGD attack
- Standard teacher-student knowledge distillation
- Pure adversarial training baseline
- Adversarial knowledge distillation, the proposed method
- Accuracy-vs-epsilon plots
- Example adversarial image visualization
- CSV output for final report tables

## Recommended Experiment Order

1. Run baseline CNN.
2. Evaluate baseline under FGSM and PGD.
3. Train teacher CNN.
4. Train standard distilled student.
5. Train pure adversarial training baseline.
6. Train adversarially distilled student.
7. Compare clean and adversarial accuracy.

## Setup

On Colab, install dependencies with:

```bash
pip install -r requirements.txt
```

On a local machine, use:

```bash
python -m pip install -r requirements.txt
```

## Quick Run

For a fast smoke test:

```bash
python src/adversarial_kd_mnist.py --epochs 1 --batch-size 128 --quick-test
```

For a fuller experiment:

```bash
python src/adversarial_kd_mnist.py --epochs 5 --batch-size 128
```

For the recommended final experiment:

```bash
python src/adversarial_kd_mnist.py --epochs 15 --batch-size 128 --pgd-steps 40 --pgd-step-size 0.01 --run-name full_e15_pgd40_advtrain
```

Outputs are written to:

```text
outputs/run_YYYYMMDD_HHMMSS/
```

Each run gets its own timestamped folder, so new experiments do not overwrite older results.

To give a run a readable name:

```bash
python src/adversarial_kd_mnist.py --epochs 5 --batch-size 128 --run-name full_e5_pgd10
```

This creates a folder like:

```text
outputs/full_e5_pgd10_YYYYMMDD_HHMMSS/
```

Each run folder contains:

```text
attack_results.csv
training_losses.csv
training_loss_curves.png
fgsm_accuracy_vs_epsilon.png
pgd_accuracy_vs_epsilon.png
fgsm_adversarial_examples.png
teacher.pt
baseline.pt
standard_kd.pt
adv_training.pt
adv_kd.pt
config.json
```

## Main Models Compared

| Model | Meaning |
|---|---|
| baseline | Ordinary CNN trained with cross-entropy |
| standard_kd | Student trained from teacher soft labels on clean images |
| adv_training | CNN trained on clean and adversarial images with hard labels only |
| adv_kd | Proposed method: student trained with clean and adversarial examples plus teacher soft labels |

## Suggested Report Question

Use the results to answer:

> Does adversarial knowledge distillation improve robustness compared with ordinary training, standard distillation, and pure adversarial training, and what clean-accuracy tradeoff does it introduce?
