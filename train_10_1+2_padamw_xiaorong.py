# -*- coding: utf-8 -*-
"""
CIFAR-10 消融实验脚本
4组:
1) AdaBelief
2) AdaBelief + 创新点1
3) AdaBelief + 创新点2
4) AdaBelief + 创新点1 + 创新点2 (MergedAdaBelief)

说明：
1. 请先把你的优化器文件名改成合法模块名，例如：
   improved_adabelief_padamw.py
   注意：不要带空格、括号、横杠

2. 并确保其中可以 import:
   - AdaBeliefInnovation1
   - AdaBeliefInnovation2
   - MergedAdaBelief

3. 本脚本不内嵌你的优化器算法，只负责训练与实验流程。

作者建议配置：
- SEED = 152
- epochs = 100
- batch_size = 512
- ResNet20
- CosineAnnealingLR
"""

import os
import csv
import time
import math
import random
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms

# ===== 你的优化器 import =====
# 1) 标准 AdaBelief
from adabelief_pytorch import AdaBelief

# 2) 你的消融版本（请保证你的优化器模块里有这几个类）
from improved_adabelief_padamw import (
    AdaBeliefInnovation1,
    AdaBeliefInnovation2,
    MergedAdaBelief,
)


# =========================
# 全局配置
# =========================
SEED = 152
NUM_CLASSES = 10
EPOCHS = 100

TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 512

LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 1e-4
AMSGRAD = False
WEIGHT_DECOUPLE = True

NUM_WORKERS = 8
PIN_MEMORY = True

CSV_DIR = "csv_save"
IMG_DIR = "image_save"

CSV_NAME = "train_cifar10_padamw_xiaorong.csv"
ACC_FIG_NAME = "train_cifar10_acc_padamw_xiaorong.png"
LOSS_FIG_NAME = "train_cifar10_loss_padamw_xiaorong.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 固定随机种子
# =========================
def set_seed(seed: int = 152):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了更强复现性，优先确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# ResNet20 for CIFAR
# =========================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet20(num_classes=10):
    # CIFAR版 ResNet20 -> 每层3个block
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


# =========================
# 参数统计
# =========================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# 数据集
# =========================
def build_dataloaders(seed=152):
    # CIFAR-10 标准归一化
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    # 验证集单独用 test_transform，避免随机增强影响验证稳定性
    full_train_dataset_for_val = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=test_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_size = 45000
    val_size = 5000
    generator = torch.Generator().manual_seed(seed)

    train_indices, val_indices = random_split(
        range(len(full_train_dataset)),
        [train_size, val_size],
        generator=generator
    )

    # 用相同索引分别构造 train / val 子集
    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(full_train_dataset_for_val, val_indices.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    return train_loader, val_loader, test_loader


# =========================
# 优化器构建
# =========================
def build_optimizer(model, optimizer_name):
    params = model.parameters()

    if optimizer_name == "AdaBelief":
        optimizer = AdaBelief(
            params,
            lr=LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            weight_decouple=WEIGHT_DECOUPLE,
            rectify=False,
            print_change_log=False,
        )

    elif optimizer_name == "AdaBelief+Innovation1":
        optimizer = AdaBeliefInnovation1(
            params,
            lr=LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            weight_decouple=WEIGHT_DECOUPLE,
            amsgrad=AMSGRAD,
        )

    elif optimizer_name == "AdaBelief+Innovation2":
        optimizer = AdaBeliefInnovation2(
            params,
            lr=LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            weight_decouple=WEIGHT_DECOUPLE,
            amsgrad=AMSGRAD,
        )

    elif optimizer_name == "AdaBelief+Innovation1+Innovation2":
        optimizer = MergedAdaBelief(
            params,
            lr=LR,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            weight_decouple=WEIGHT_DECOUPLE,
            p_adapt=0.497,
            amsgrad=AMSGRAD,
        )

    else:
        raise ValueError(f"未知优化器名称: {optimizer_name}")

    return optimizer


# =========================
# 单轮训练 / 评估
# =========================
def run_one_epoch_train(model, loader, criterion, optimizer, scaler, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def run_one_epoch_eval(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


# =========================
# 绘图
# =========================
def plot_curves(history, save_path, metric="acc"):
    """
    history:
    {
        optimizer_name: {
            "train_acc": [...],
            "val_acc": [...],
            "test_acc": [...],
            "train_loss": [...],
            "val_loss": [...],
            "test_loss": [...]
        }
    }
    """
    plt.figure(figsize=(14, 8))

    for opt_name, records in history.items():
        if metric == "acc":
            plt.plot(records["train_acc"], label=f"{opt_name}-Train")
            plt.plot(records["val_acc"], linestyle="--", label=f"{opt_name}-Val")
            plt.plot(records["test_acc"], linestyle=":", label=f"{opt_name}-Test")
            plt.ylabel("Accuracy (%)")
            plt.title("CIFAR-10 Ablation Accuracy Curves")
        else:
            plt.plot(records["train_loss"], label=f"{opt_name}-Train")
            plt.plot(records["val_loss"], linestyle="--", label=f"{opt_name}-Val")
            plt.plot(records["test_loss"], linestyle=":", label=f"{opt_name}-Test")
            plt.ylabel("Loss")
            plt.title("CIFAR-10 Ablation Loss Curves")

    plt.xlabel("Epoch")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# CSV 保存
# =========================
def save_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "seed",
        "optimizer",
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "epoch_time_sec",
        "total_time_sec",
    ]

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 单个优化器实验
# =========================
def run_experiment(optimizer_name, train_loader, val_loader, test_loader):
    print("\n" + "=" * 90)
    print(f"开始实验: {optimizer_name}")
    print("=" * 90)

    model = ResNet20(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    n_params = count_parameters(model)
    print(f"Using device: {DEVICE}")
    print(f"Model: ResNet20")
    print(f"Model Parameters: {n_params / 1e6:.3f} M")
    print(f"Seed: {SEED}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"Eval batch size: {EVAL_BATCH_SIZE}")
    print(f"Initial LR: {LR}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Scheduler: CosineAnnealingLR")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    csv_rows = []
    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss, train_acc = run_one_epoch_train(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        val_loss, val_acc = run_one_epoch_eval(
            model, val_loader, criterion, DEVICE
        )
        test_loss, test_acc = run_one_epoch_eval(
            model, test_loader, criterion, DEVICE
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        csv_rows.append({
            "seed": SEED,
            "optimizer": optimizer_name,
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time_sec": epoch_time,
            "total_time_sec": total_time,
        })

        print(
            f"[{optimizer_name}] "
            f"Epoch [{epoch:03d}/{EPOCHS:03d}] | "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
            f"Epoch Time: {epoch_time:.2f}s | "
            f"Total Time: {total_time:.2f}s"
        )

    print(f"\n实验结束: {optimizer_name}")
    print(f"Final Train Acc: {history['train_acc'][-1]:.2f}% | Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val   Acc: {history['val_acc'][-1]:.2f}% | Final Val   Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Test  Acc: {history['test_acc'][-1]:.2f}% | Final Test  Loss: {history['test_loss'][-1]:.4f}")
    print(f"总耗时: {time.time() - total_start_time:.2f}s")

    return history, csv_rows


# =========================
# 主函数
# =========================
def main():
    warnings.filterwarnings("ignore")
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    set_seed(SEED)

    print("=" * 90)
    print("CIFAR-10 消融实验开始")
    print("=" * 90)

    train_loader, val_loader, test_loader = build_dataloaders(seed=SEED)

    optimizer_list = [
        "AdaBelief",
        "AdaBelief+Innovation1",
        "AdaBelief+Innovation2",
        "AdaBelief+Innovation1+Innovation2",
    ]

    all_history = {}
    all_csv_rows = []

    overall_start = time.time()

    for optimizer_name in optimizer_list:
        history, csv_rows = run_experiment(
            optimizer_name,
            train_loader,
            val_loader,
            test_loader
        )
        all_history[optimizer_name] = history
        all_csv_rows.extend(csv_rows)

    csv_path = os.path.join(CSV_DIR, CSV_NAME)
    acc_fig_path = os.path.join(IMG_DIR, ACC_FIG_NAME)
    loss_fig_path = os.path.join(IMG_DIR, LOSS_FIG_NAME)

    save_csv(all_csv_rows, csv_path)
    plot_curves(all_history, acc_fig_path, metric="acc")
    plot_curves(all_history, loss_fig_path, metric="loss")

    total_time = time.time() - overall_start

    print("\n" + "=" * 90)
    print("所有实验完成")
    print("=" * 90)
    print(f"CSV 已保存到: {csv_path}")
    print(f"ACC 图已保存到: {acc_fig_path}")
    print(f"LOSS 图已保存到: {loss_fig_path}")
    print(f"总耗时: {total_time:.2f}s")


if __name__ == "__main__":
    # Windows 下多进程 DataLoader 需要这个入口保护
    main()