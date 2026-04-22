"""MNIST adversarial robustness framework.

This script supports the final-project pipeline:

1. Train a baseline CNN.
2. Generate FGSM and PGD adversarial examples.
3. Train a teacher model.
4. Train a standard distilled student.
5. Train the proposed adversarially distilled student.
6. Compare all models with accuracy-vs-epsilon curves.

The proposed method is adversarial knowledge distillation: the student learns
from both clean and adversarial images while combining hard-label supervision
with teacher soft-label supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


EPSILONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


@dataclass
class Config:
    data_dir: str = "data"
    output_dir: str = "outputs"
    run_name: str | None = None
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    temperature: float = 5.0
    kd_alpha: float = 0.5
    adv_weight: float = 0.5
    pgd_steps: int = 10
    pgd_step_size: float = 0.01
    adv_train_epsilon: float = 0.1
    quick_test: bool = False


class SmallCNN(nn.Module):
    """Compact CNN suitable for MNIST and fast experiments."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="MNIST adversarial KD framework")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--kd-alpha", type=float, default=0.5)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-step-size", type=float, default=0.01)
    parser.add_argument("--adv-train-epsilon", type=float, default=0.1)
    parser.add_argument("--quick-test", action="store_true")
    return Config(**vars(parser.parse_args()))


def make_run_dir(config: Config) -> Path:
    """Create a fresh output directory for every experiment run."""

    root = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.run_name:
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in config.run_name)
        run_dir = root / f"{safe_name}_{timestamp}"
    else:
        run_dir = root / f"run_{timestamp}"

    suffix = 1
    candidate = run_dir
    while candidate.exists():
        candidate = Path(f"{run_dir}_{suffix}")
        suffix += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)

    if config.quick_test:
        train_set = Subset(train_set, range(2048))
        test_set = Subset(test_set, range(1024))

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if epsilon == 0:
        return x.detach()

    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + epsilon * grad.sign()
    return torch.clamp(x_adv, 0.0, 1.0).detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    if epsilon == 0:
        return x.detach()

    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + step_size * grad.sign()
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv


def evaluate_under_attack(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, float], torch.Tensor],
    epsilons: list[float],
) -> dict[float, float]:
    model.eval()
    results = {}

    for epsilon in epsilons:
        correct = 0
        total = 0
        for x, y in tqdm(loader, desc=f"eval eps={epsilon}", leave=False):
            x = x.to(device)
            y = y.to(device)
            x_adv = attack_fn(model, x, y, epsilon)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        results[epsilon] = correct / total

    return results


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    name: str,
) -> list[float]:
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"{name} epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            count += y.size(0)
        losses.append(running / count)
        print(f"{name}: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def train_standard_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    optimizer = optim.Adam(student.parameters(), lr=config.lr)
    teacher.eval()
    losses = []

    for epoch in range(1, config.epochs + 1):
        student.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"standard_kd epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)
            hard_loss = F.cross_entropy(student_logits, y)
            soft_loss = kd_loss(student_logits, teacher_logits, config.temperature)
            loss = (1 - config.kd_alpha) * hard_loss + config.kd_alpha * soft_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            count += y.size(0)

        losses.append(running / count)
        print(f"standard_kd: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def train_adversarial(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    """Train on clean and adversarial images using only hard labels."""

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"adv_training epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            model.eval()
            x_adv = fgsm_attack(model, x, y, config.adv_train_epsilon)
            model.train()

            clean_logits = model(x)
            adv_logits = model(x_adv)

            clean_loss = F.cross_entropy(clean_logits, y)
            adv_loss = F.cross_entropy(adv_logits, y)
            loss = (1 - config.adv_weight) * clean_loss + config.adv_weight * adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            count += y.size(0)

        losses.append(running / count)
        print(f"adv_training: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def train_adversarial_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> list[float]:
    """Train the proposed method: KD on clean and adversarial images."""

    optimizer = optim.Adam(student.parameters(), lr=config.lr)
    teacher.eval()
    losses = []

    for epoch in range(1, config.epochs + 1):
        student.train()
        running = 0.0
        count = 0
        for x, y in tqdm(loader, desc=f"adv_kd epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)

            student.eval()
            x_adv = fgsm_attack(student, x, y, config.adv_train_epsilon)
            student.train()

            with torch.no_grad():
                teacher_clean_logits = teacher(x)
                teacher_adv_logits = teacher(x_adv)

            clean_logits = student(x)
            adv_logits = student(x_adv)

            clean_hard = F.cross_entropy(clean_logits, y)
            clean_soft = kd_loss(clean_logits, teacher_clean_logits, config.temperature)
            adv_hard = F.cross_entropy(adv_logits, y)
            adv_soft = kd_loss(adv_logits, teacher_adv_logits, config.temperature)

            clean_loss = (1 - config.kd_alpha) * clean_hard + config.kd_alpha * clean_soft
            adv_loss = (1 - config.kd_alpha) * adv_hard + config.kd_alpha * adv_soft
            loss = (1 - config.adv_weight) * clean_loss + config.adv_weight * adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * y.size(0)
            count += y.size(0)

        losses.append(running / count)
        print(f"adv_kd: epoch {epoch}/{config.epochs}, loss={losses[-1]:.4f}")

    return losses


def plot_curves(
    results: dict[str, dict[str, dict[float, float]]],
    output_dir: Path,
) -> None:
    for attack_name in ["fgsm", "pgd"]:
        plt.figure(figsize=(7, 5))
        for model_name, model_results in results.items():
            attack_results = model_results[attack_name]
            xs = list(attack_results.keys())
            ys = [attack_results[e] for e in xs]
            plt.plot(xs, ys, marker="o", label=model_name)
        plt.xlabel("epsilon")
        plt.ylabel("accuracy")
        plt.title(f"MNIST accuracy under {attack_name.upper()} attack")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{attack_name}_accuracy_vs_epsilon.png", dpi=200)
        plt.close()


def save_training_losses_csv(
    losses_by_model: dict[str, list[float]],
    output_dir: Path,
) -> None:
    path = output_dir / "training_losses.csv"
    max_epochs = max(len(losses) for losses in losses_by_model.values())
    model_names = list(losses_by_model.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *model_names])
        for epoch_idx in range(max_epochs):
            row = [epoch_idx + 1]
            for model_name in model_names:
                losses = losses_by_model[model_name]
                row.append(losses[epoch_idx] if epoch_idx < len(losses) else "")
            writer.writerow(row)


def plot_training_losses(
    losses_by_model: dict[str, list[float]],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    for model_name, losses in losses_by_model.items():
        xs = list(range(1, len(losses) + 1))
        plt.plot(xs, losses, marker="o", label=model_name)
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Training loss by model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss_curves.png", dpi=200)
    plt.close()


def save_results_csv(
    results: dict[str, dict[str, dict[float, float]]],
    output_dir: Path,
) -> None:
    path = output_dir / "attack_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "attack", "epsilon", "accuracy"])
        for model_name, model_results in results.items():
            for attack_name, attack_results in model_results.items():
                for epsilon, acc in attack_results.items():
                    writer.writerow([model_name, attack_name, epsilon, acc])


def visualize_adversarial_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Config,
    output_dir: Path,
) -> None:
    model.eval()
    x, y = next(iter(loader))
    x = x[:8].to(device)
    y = y[:8].to(device)
    x_adv = fgsm_attack(model, x, y, config.adv_train_epsilon)

    with torch.no_grad():
        clean_pred = model(x).argmax(dim=1)
        adv_pred = model(x_adv).argmax(dim=1)

    plt.figure(figsize=(10, 4))
    for i in range(x.size(0)):
        plt.subplot(2, x.size(0), i + 1)
        plt.imshow(x[i].detach().cpu().squeeze(), cmap="gray")
        plt.title(f"y={y[i].item()}\np={clean_pred[i].item()}")
        plt.axis("off")

        plt.subplot(2, x.size(0), x.size(0) + i + 1)
        plt.imshow(x_adv[i].detach().cpu().squeeze(), cmap="gray")
        plt.title(f"adv p={adv_pred[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "fgsm_adversarial_examples.png", dpi=200)
    plt.close()


def save_checkpoint(model: nn.Module, output_dir: Path, name: str) -> None:
    torch.save(model.state_dict(), output_dir / f"{name}.pt")


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    output_dir = make_run_dir(config)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Saving this run to: {output_dir.resolve()}")
    train_loader, test_loader = get_loaders(config)

    teacher = SmallCNN(dropout=0.25).to(device)
    baseline = SmallCNN(dropout=0.25).to(device)
    standard_kd = SmallCNN(dropout=0.25).to(device)
    adv_training = SmallCNN(dropout=0.25).to(device)
    adv_kd = SmallCNN(dropout=0.25).to(device)

    losses_by_model: dict[str, list[float]] = {}

    print("\nTraining teacher")
    losses_by_model["teacher"] = train_supervised(teacher, train_loader, device, config, name="teacher")
    save_checkpoint(teacher, output_dir, "teacher")

    print("\nTraining baseline")
    losses_by_model["baseline"] = train_supervised(baseline, train_loader, device, config, name="baseline")
    save_checkpoint(baseline, output_dir, "baseline")

    print("\nTraining standard KD student")
    losses_by_model["standard_kd"] = train_standard_kd(standard_kd, teacher, train_loader, device, config)
    save_checkpoint(standard_kd, output_dir, "standard_kd")

    print("\nTraining adversarial training baseline")
    losses_by_model["adv_training"] = train_adversarial(adv_training, train_loader, device, config)
    save_checkpoint(adv_training, output_dir, "adv_training")

    print("\nTraining adversarial KD student")
    losses_by_model["adv_kd"] = train_adversarial_kd(adv_kd, teacher, train_loader, device, config)
    save_checkpoint(adv_kd, output_dir, "adv_kd")
    save_training_losses_csv(losses_by_model, output_dir)
    plot_training_losses(losses_by_model, output_dir)

    models = {
        "baseline": baseline,
        "standard_kd": standard_kd,
        "adv_training": adv_training,
        "adv_kd": adv_kd,
    }

    pgd_fn = lambda m, x, y, eps: pgd_attack(
        m,
        x,
        y,
        epsilon=eps,
        steps=config.pgd_steps,
        step_size=config.pgd_step_size,
    )

    results: dict[str, dict[str, dict[float, float]]] = {}
    for model_name, model in models.items():
        clean_acc = accuracy(model, test_loader, device)
        print(f"{model_name}: clean accuracy={clean_acc:.4f}")
        results[model_name] = {
            "fgsm": evaluate_under_attack(model, test_loader, device, fgsm_attack, EPSILONS),
            "pgd": evaluate_under_attack(model, test_loader, device, pgd_fn, EPSILONS),
        }

    save_results_csv(results, output_dir)
    plot_curves(results, output_dir)
    visualize_adversarial_examples(baseline, test_loader, device, config, output_dir)

    print(f"\nDone. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
