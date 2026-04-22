# Experiment Plan

## Project Question

Can a student model become more robust to adversarial attacks if it learns from both teacher soft labels and adversarial examples?

## Models

### 1. Baseline CNN

Train a normal CNN on MNIST with cross-entropy loss.

Purpose:

- Shows normal clean-image performance.
- Gives a reference point for adversarial vulnerability.

### 2. Standard Knowledge Distillation

Train a teacher CNN first. Then train a student CNN using:

```text
loss = hard-label cross entropy + teacher soft-label distillation loss
```

Purpose:

- Tests whether ordinary distillation improves robustness.
- This is a comparison method, not the main innovation.

### 3. Pure Adversarial Training

Train a CNN using both clean images and adversarial images, but only with hard labels:

```text
clean loss = CE(model(clean), label)
adv loss   = CE(model(adv), label)

final loss = clean loss + adversarial loss
```

Purpose:

- Tests how much robustness comes from adversarial examples alone.
- Provides a key comparison for the proposed method.

### 4. Adversarial Knowledge Distillation

This is the proposed method.

Train the student using both clean images and adversarial images:

```text
clean loss = CE(student(clean), label) + KD(student(clean), teacher(clean))
adv loss   = CE(student(adv), label)   + KD(student(adv), teacher(adv))

final loss = clean loss + adversarial loss
```

Purpose:

- Combines knowledge distillation and adversarial training.
- Goes beyond simply implementing FGSM/PGD and standard distillation.

## Attacks

### FGSM

One-step gradient attack. It is fast and useful as a first robustness test.

### PGD

Multi-step projected gradient attack. It is stronger than FGSM and gives a more serious robustness test.

## Epsilon Values

Use:

```text
0, 0.01, 0.02, 0.05, 0.1, 0.2
```

Interpretation:

- `epsilon = 0` means clean test accuracy.
- Larger epsilon means stronger attack.

## Figures to Include

1. Clean accuracy table.
2. FGSM accuracy vs epsilon curve.
3. PGD accuracy vs epsilon curve.
4. Example adversarial images.
5. Optional: table comparing robustness at `epsilon = 0.1`.

## Main Discussion Points

Answer these in the report:

1. Does standard distillation improve robustness over the baseline?
2. Does pure adversarial training improve robustness over the baseline?
3. Does adversarial KD improve robustness over pure adversarial training?
4. Is there a tradeoff between clean accuracy and adversarial accuracy?
5. Is PGD harder to defend against than FGSM?
6. What are the limitations of the proposed method?

## Expected Presentation Story

1. CNNs can classify clean MNIST well.
2. Small adversarial perturbations can fool them.
3. Standard knowledge distillation may smooth student predictions, but it is not enough by itself.
4. Pure adversarial training shows what adversarial examples contribute without distillation.
5. We propose adversarial knowledge distillation.
6. We compare all models under FGSM and PGD.
7. We discuss whether the proposed method improves robustness and what tradeoffs appear.
