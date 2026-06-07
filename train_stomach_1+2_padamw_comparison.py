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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ========== 直接导入你的算法 ==========
from improved_adabelief_padamw import MergedAdaBelief

# ========== AdaBelief ==========
from adabelief_pytorch import AdaBelief


# =========================================================
# 1. 全局配置
# =========================================================
SEED = 152
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 4           # Windows 下更稳；如仍有多进程问题可改为 0
IMAGE_SIZE = 128

DATA_ROOT = r"C:\Users\admin\Desktop\syzhuomian\data\stomach"

CSV_DIR = "csv_save"
IMG_DIR = "image_save"

CSV_PATH = os.path.join(CSV_DIR, "train_stomach_1+2_padamw_comparison11.csv")
ACC_FIG_PATH = os.path.join(IMG_DIR, "train_stomach_acc_1+2_padamw_comparison11.png")
LOSS_FIG_PATH = os.path.join(IMG_DIR, "train_stomach_loss_1+2_padamw_comparison11.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTIMIZER_LIST = [
    "AdaBelief",
    "MergedAdaBelief",
    "Adam",
    "AdamW",
    "AdaGrad",
    "NAdam",
]

# 固定划分数量（总计 1885）
TRAIN_TOTAL = 900
VAL_TOTAL = 500
TEST_TOTAL = 485


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
# 3. 更适合病理图像与自适应优化器的网络
#    PathologyResNetSE: 残差 + SE + GELU + 深一点的层级结构
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvBNGELU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, reduction=8, drop_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction=reduction)
        self.drop = nn.Dropout2d(drop_p) if drop_p > 0 else nn.Identity()
        self.act2 = nn.GELU()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.drop(out)

        out = out + identity
        out = self.act2(out)
        return out


class PathologyResNetSE(nn.Module):
    def __init__(self, num_classes=8, base_width=32):
        super().__init__()

        # stem：尽量保留病理纹理信息，不用过强的早期池化
        self.stem = nn.Sequential(
            ConvBNGELU(3, base_width, kernel_size=3, stride=1),
            ConvBNGELU(base_width, base_width, kernel_size=3, stride=1),
        )

        self.layer1 = nn.Sequential(
            ResidualSEBlock(base_width, base_width, stride=1, reduction=8, drop_p=0.00),
            ResidualSEBlock(base_width, base_width, stride=1, reduction=8, drop_p=0.00),
        )

        self.layer2 = nn.Sequential(
            ResidualSEBlock(base_width, base_width * 2, stride=2, reduction=8, drop_p=0.03),
            ResidualSEBlock(base_width * 2, base_width * 2, stride=1, reduction=8, drop_p=0.03),
        )

        self.layer3 = nn.Sequential(
            ResidualSEBlock(base_width * 2, base_width * 4, stride=2, reduction=8, drop_p=0.05),
            ResidualSEBlock(base_width * 4, base_width * 4, stride=1, reduction=8, drop_p=0.05),
        )

        self.layer4 = nn.Sequential(
            ResidualSEBlock(base_width * 4, base_width * 6, stride=2, reduction=8, drop_p=0.08),
            ResidualSEBlock(base_width * 6, base_width * 6, stride=1, reduction=8, drop_p=0.08),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_width * 6, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)   # 128x128
        x = self.layer2(x)   # 64x64
        x = self.layer3(x)   # 32x32
        x = self.layer4(x)   # 16x16
        x = self.head(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 4. 分层抽样划分
# =========================================================
def stratified_split_indices(targets, train_total, val_total, test_total, seed=152):
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    num_samples = len(targets)

    if train_total + val_total + test_total != num_samples:
        raise ValueError(
            f"train_total + val_total + test_total != 数据总数: "
            f"{train_total}+{val_total}+{test_total} != {num_samples}"
        )

    classes = np.unique(targets)
    class_to_indices = {}
    class_counts = {}

    for cls in classes:
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        class_to_indices[cls] = idx
        class_counts[cls] = len(idx)

    def allocate(total_needed):
        raw = {cls: class_counts[cls] * total_needed / num_samples for cls in classes}
        base = {cls: int(np.floor(raw[cls])) for cls in classes}
        remain = total_needed - sum(base.values())

        frac_sorted = sorted(classes, key=lambda c: (raw[c] - base[c]), reverse=True)
        for i in range(remain):
            base[frac_sorted[i]] += 1
        return base

    train_quota = allocate(train_total)
    val_quota = allocate(val_total)
    test_quota = {
        cls: class_counts[cls] - train_quota[cls] - val_quota[cls]
        for cls in classes
    }

    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        idx = class_to_indices[cls]
        n_train = train_quota[cls]
        n_val = val_quota[cls]
        n_test = test_quota[cls]

        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:n_train + n_val + n_test].tolist())

    return train_idx, val_idx, test_idx, train_quota, val_quota, test_quota


# =========================================================
# 5. 数据集
# =========================================================
def get_dataloaders(batch_size=64, num_workers=4):
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"未找到数据集目录: {DATA_ROOT}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    full_dataset_eval = datasets.ImageFolder(DATA_ROOT, transform=eval_transform)
    full_dataset_train = datasets.ImageFolder(DATA_ROOT, transform=train_transform)

    if full_dataset_eval.class_to_idx != full_dataset_train.class_to_idx:
        raise ValueError("训练集和评估集的 class_to_idx 不一致。")

    targets = full_dataset_eval.targets
    class_names = full_dataset_eval.classes

    train_idx, val_idx, test_idx, train_quota, val_quota, test_quota = stratified_split_indices(
        targets=targets,
        train_total=TRAIN_TOTAL,
        val_total=VAL_TOTAL,
        test_total=TEST_TOTAL,
        seed=SEED
    )

    train_dataset = Subset(full_dataset_train, train_idx)
    val_dataset = Subset(full_dataset_eval, val_idx)
    test_dataset = Subset(full_dataset_eval, test_idx)

    loader_generator = get_torch_generator(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=loader_generator
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=loader_generator
    )

    return train_loader, val_loader, test_loader, class_names, train_quota, val_quota, test_quota


# =========================================================
# 6. 训练与评估
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
# 7. 构建优化器
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
# 8. 单个实验
# =========================================================
def run_experiment(optimizer_name, train_loader, val_loader, test_loader, device, num_classes):
    print(f"\n{'=' * 20} {optimizer_name} {'=' * 20}")

    set_seed(SEED)

    model = PathologyResNetSE(num_classes=num_classes, base_width=32).to(device)
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
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - total_start
    print(f"{optimizer_name} finished. Total Time: {total_time / 60:.2f} min")
    print(f"{optimizer_name} Best Val Acc: {best_val_acc:.2f}%")
    print(f"{optimizer_name} Test Acc @ Best Val: {best_test_acc_at_best_val:.2f}%")

    return results


# =========================================================
# 9. 保存 CSV
# =========================================================
def save_results_to_csv(all_results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已保存到: {csv_path}")


# =========================================================
# 10. 绘图
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
    plt.title("stomach Train and Validation Accuracy Comparison (PathologyResNetSE)")
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
    plt.title("stomach Train and Validation Loss Comparison (PathologyResNetSE)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()
    print(f"损失图已保存到: {loss_fig_path}")


# =========================================================
# 11. 主函数
# =========================================================
def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    print("Using device:", DEVICE)
    print("Setting seed =", SEED)
    print("Dataset root:", DATA_ROOT)
    set_seed(SEED)

    train_loader, val_loader, test_loader, class_names, train_quota, val_quota, test_quota = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print("Classes:", class_names)
    print("Number of classes:", len(class_names))

    print("\n===== Stratified Split Summary =====")
    for cls_name in class_names:
        cls_idx = class_names.index(cls_name)
        print(
            f"{cls_name}: "
            f"train={train_quota[cls_idx]}, "
            f"val={val_quota[cls_idx]}, "
            f"test={test_quota[cls_idx]}"
        )

    print(
        f"Total: train={sum(train_quota.values())}, "
        f"val={sum(val_quota.values())}, "
        f"test={sum(test_quota.values())}"
    )

    temp_model = PathologyResNetSE(num_classes=len(class_names), base_width=32)
    total_params = count_parameters(temp_model)
    print(f"Model Parameters: {total_params / 1e6:.3f} M")

    all_results = []

    for optimizer_name in OPTIMIZER_LIST:
        results = run_experiment(
            optimizer_name=optimizer_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            num_classes=len(class_names),
        )
        all_results.extend(results)

    save_results_to_csv(all_results, CSV_PATH)
    plot_curves(CSV_PATH, ACC_FIG_PATH, LOSS_FIG_PATH)

    print("\n全部实验完成。")


if __name__ == "__main__":
    main()
