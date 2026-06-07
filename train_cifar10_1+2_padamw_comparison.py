import os

# =========================================================
# 必须放在 import torch 之前（严格复现 + CUDA/cuBLAS）
# =========================================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "152"

import time
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms

# ========== 直接导入你的算法 ==========
from improved_adabelief_padamw import MergedAdaBelief

# ========== AdaBelief ==========
# pip install adabelief-pytorch
from adabelief_pytorch import AdaBelief


# =========================================================
# 1. 全局配置
# =========================================================
SEED = 152
NUM_EPOCHS = 100
BATCH_SIZE = 256
NUM_WORKERS = 8
NUM_CLASSES = 10

CSV_DIR = "csv_save"
IMG_DIR = "image_save"

CSV_PATH = os.path.join(CSV_DIR, "train_cifar10_padamw_comparison.csv")
ACC_FIG_PATH = os.path.join(IMG_DIR, "train_cifar10_acc_padamw_comparison.png")
LOSS_FIG_PATH = os.path.join(IMG_DIR, "train_cifar10_loss_padamw_comparison.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTIMIZER_LIST = [
    "AdaBelief",
    "MergedAdaBelief",
    "Adam",
    "AdamW",
    "AdaGrad",
    "NAdam",
]


# =========================================================
# 2. 严格复现
# =========================================================
def set_seed(seed: int = 152):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_torch_generator(seed=152):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# =========================================================
# 3. CIFAR 风格 ResNet20
# =========================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 4. 数据集
# =========================================================
def get_dataloaders(seed=152, batch_size=256, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    full_train_dataset_aug = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    full_train_dataset_eval = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=eval_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=eval_transform
    )

    generator_split = get_torch_generator(seed)

    train_size = 45000
    val_size = 5000

    train_subset_tmp, val_subset_tmp = random_split(
        full_train_dataset_aug,
        [train_size, val_size],
        generator=generator_split
    )

    train_indices = train_subset_tmp.indices
    val_indices = val_subset_tmp.indices

    train_dataset = Subset(full_train_dataset_aug, train_indices)
    val_dataset = Subset(full_train_dataset_eval, val_indices)

    loader_generator = get_torch_generator(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=loader_generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=loader_generator
    )

    return train_loader, val_loader, test_loader


# =========================================================
# 5. 训练与评估
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = 100.0 * total_correct / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = 100.0 * total_correct / total_samples
    return epoch_loss, epoch_acc


# =========================================================
# 6. 构建优化器
# =========================================================
def build_optimizer(model, optimizer_name):
    name = optimizer_name.lower()

    if name == "adabelief":
        return AdaBelief(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            weight_decouple=True,
            rectify=False,
            print_change_log=False
        )

    elif name == "mergedadabelief":
        return MergedAdaBelief(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            weight_decouple=True,
            amsgrad=False
        )

    elif name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )

    elif name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )

    elif name == "adagrad":
        return optim.Adagrad(
            model.parameters(),
            lr=1e-2,
            eps=1e-10,
            weight_decay=1e-4
        )

    elif name == "nadam":
        return optim.NAdam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4
        )

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# =========================================================
# 7. 单个实验
# =========================================================
def run_experiment(optimizer_name, train_loader, val_loader, test_loader, device):
    print(f"\n{'=' * 20} {optimizer_name} {'=' * 20}")

    set_seed(SEED)

    model = ResNet20(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, optimizer_name)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )

    results = []
    best_val_acc = -1.0
    best_test_acc_at_best_val = -1.0

    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc_at_best_val = test_acc

        row = OrderedDict({
            "Optimizer": optimizer_name,
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Acc": train_acc,
            "Val Loss": val_loss,
            "Val Acc": val_acc,
            "Test Loss": test_loss,
            "Test Acc": test_acc,
            "LR": current_lr,
            "Time(s)": epoch_time
        })
        results.append(row)

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] | "
            f"Optimizer: {optimizer_name} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - total_start
    print(f"{optimizer_name} finished. Total Time: {total_time / 60:.2f} min")
    print(f"{optimizer_name} Best Val Acc: {best_val_acc:.2f}%")
    print(f"{optimizer_name} Test Acc @ Best Val: {best_test_acc_at_best_val:.2f}%")

    return results


# =========================================================
# 8. 保存 CSV
# =========================================================
def save_results_to_csv(all_results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已保存到: {csv_path}")


# =========================================================
# 9. 绘图（画 Train / Val）
# =========================================================
def plot_curves(csv_path, acc_fig_path, loss_fig_path):
    os.makedirs(os.path.dirname(acc_fig_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_fig_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(14, 8))
    for optimizer_name in df["Optimizer"].unique():
        sub_df = df[df["Optimizer"] == optimizer_name]
        plt.plot(sub_df["Epoch"], sub_df["Train Acc"], label=f"{optimizer_name} Train Acc")
        plt.plot(sub_df["Epoch"], sub_df["Val Acc"], linestyle="--", label=f"{optimizer_name} Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CIFAR-10 Train and Validation Accuracy Comparison")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(acc_fig_path, dpi=300)
    plt.close()
    print(f"准确率图已保存到: {acc_fig_path}")

    plt.figure(figsize=(14, 8))
    for optimizer_name in df["Optimizer"].unique():
        sub_df = df[df["Optimizer"] == optimizer_name]
        plt.plot(sub_df["Epoch"], sub_df["Train Loss"], label=f"{optimizer_name} Train Loss")
        plt.plot(sub_df["Epoch"], sub_df["Val Loss"], linestyle="--", label=f"{optimizer_name} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CIFAR-10 Train and Validation Loss Comparison")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()
    print(f"损失图已保存到: {loss_fig_path}")


# =========================================================
# 10. 主函数
# =========================================================
def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Using device:", DEVICE)
    print("Setting seed =", SEED)
    set_seed(SEED)

    train_loader, val_loader, test_loader = get_dataloaders(
        seed=SEED,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    temp_model = ResNet20(num_classes=NUM_CLASSES)
    total_params = count_parameters(temp_model)
    print(f"Model Parameters: {total_params / 1e6:.3f} M")

    all_results = []

    for optimizer_name in OPTIMIZER_LIST:
        results = run_experiment(
            optimizer_name=optimizer_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE
        )
        all_results.extend(results)

    save_results_to_csv(all_results, CSV_PATH)
    plot_curves(CSV_PATH, ACC_FIG_PATH, LOSS_FIG_PATH)

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()