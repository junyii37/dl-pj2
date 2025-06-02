import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam

from VGG_BatchNorm.models.vgg import VGG_A, VGG_A_BatchNorm
from VGG_BatchNorm.data.loaders import get_cifar_loader

# =============================================================================
# 全局配置
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4         # 若 get_cifar_loader 不使用此参数，可删除
BATCH_SIZE = 128        # 若 get_cifar_loader 不使用此参数，可删除
EPOCHS = 20             # 默认训练轮数
LR_LIST = [1e-3, 2e-3, 1e-4, 5e-4]  # 用于 Loss 稳定性实验的学习率列表

MODULE_PATH = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录（项目根）
FIGURES_PATH = os.path.join(MODULE_PATH, "reports", "figures")
MODELS_PATH = os.path.join(MODULE_PATH, "reports", "models")

# 创建保存目录（如不存在则自动创建）
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# print(f"Using device: {DEVICE}\n")


# =============================================================================
# 辅助函数
# =============================================================================
def set_random_seeds(seed: int = 0, device: torch.device = torch.device("cpu")) -> None:
    """
    设置随机种子，确保实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type != "cpu":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_accuracy(model: nn.Module,
                      data_loader: torch.utils.data.DataLoader) -> float:
    """
    计算模型在给定数据集上的分类准确率。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# =============================================================================
# 核心训练逻辑
# =============================================================================
def train_one_model(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    epochs: int = EPOCHS,
                    scheduler=None,
                    best_model_filename: str = None
                    ) -> tuple:
    """
    训练单个模型并收集指标。
    返回：
      - epoch_train_loss: 每个 epoch 的平均训练 loss (长度 = epochs)
      - epoch_train_acc: 每个 epoch 的训练集准确率 (长度 = epochs)
      - epoch_val_acc:   每个 epoch 的验证集准确率 (长度 = epochs)
      - all_step_losses: 每个 step (batch) 的 loss，列表嵌套 (shape = [epochs][batches_per_epoch])
    """
    model = model.to(DEVICE)
    model_name = model.__class__.__name__
    batches_per_epoch = len(train_loader)

    epoch_train_loss = [np.nan] * epochs
    epoch_train_acc = [np.nan] * epochs
    epoch_val_acc = [np.nan] * epochs
    all_step_losses = []

    best_val_acc = 0.0
    best_epoch = -1

    print(f"--> Training {model_name} for {epochs} epochs")
    for epoch in tqdm(range(epochs), desc=f"{model_name} Epochs", unit="ep"):
        if scheduler:
            scheduler.step()

        model.train()
        running_loss = 0.0
        step_losses = []

        # --- 逐批次训练 ---
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_losses.append(loss.item())

        # 记录本 epoch 平均训练 loss
        epoch_train_loss[epoch] = running_loss / batches_per_epoch
        all_step_losses.append(step_losses)

        # --- 评估训练集和验证集准确率 ---
        model.eval()
        with torch.no_grad():
            train_acc = evaluate_accuracy(model, train_loader)
            val_acc = evaluate_accuracy(model, val_loader)
        epoch_train_acc[epoch] = train_acc
        epoch_val_acc[epoch] = val_acc

        # 保存验证集最好模型
        if best_model_filename and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(MODELS_PATH, best_model_filename))

    if best_model_filename:
        print(f"[{model_name}] Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch + 1}\n")
    else:
        print(f"[{model_name}] Training finished (no best-model tracking)\n")

    return epoch_train_loss, epoch_train_acc, epoch_val_acc, all_step_losses


# =============================================================================
# 可视化函数
# =============================================================================
def plot_train_curves(epochs: int,
                      loss_A: list,
                      accA_train: list,
                      accA_val: list,
                      loss_BN: list,
                      accBN_train: list,
                      accBN_val: list) -> None:
    """
    同时绘制 VGG_A（无 BN）与 VGG_A_BatchNorm（有 BN）的训练曲线对比：
      - 平均训练 Loss
      - 训练/验证 Accuracy
    将图像保存到 FIGURES_PATH 目录下。
    """
    print("--> Plotting training curves comparison")
    # —— 平均训练 Loss 对比 ——
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_A, label="VGG_A (no BN) - Train Loss")
    plt.plot(range(1, epochs + 1), loss_BN, label="VGG_A_BatchNorm - Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss Comparison")
    plt.legend()
    plt.grid(True)
    fname1 = os.path.join(FIGURES_PATH, "train_loss_comparison.png")
    plt.savefig(fname1)
    plt.close()
    print(f"Saved: {fname1}")

    # —— 训练/验证 Accuracy 对比 ——
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), accA_train, "--", label="VGG_A Train Acc")
    plt.plot(range(1, epochs + 1), accA_val, "-", label="VGG_A Val Acc")
    plt.plot(range(1, epochs + 1), accBN_train, "--", label="VGG_A_BatchNorm Train Acc")
    plt.plot(range(1, epochs + 1), accBN_val, "-", label="VGG_A_BatchNorm Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    fname2 = os.path.join(FIGURES_PATH, "accuracy_comparison.png")
    plt.savefig(fname2)
    plt.close()
    print(f"Saved: {fname2}\n")


def run_loss_stability_experiment(ModelClass,
                                  lr_list: list,
                                  train_loader,
                                  val_loader,
                                  epochs: int = EPOCHS) -> tuple:
    """
    对同一模型使用不同学习率进行训练，记录每个 batch 的 loss。
    计算跨学习率的 min_curve 和 max_curve，并保存 Loss 区间图。
    返回：(min_curve, max_curve)，长度 = epochs * batches_per_epoch。
    """
    model_name = ModelClass.__name__
    all_step_losses = []

    print(f"--> Running loss stability experiment for {model_name}")
    for lr in lr_list:
        print(f"   - Learning rate: {lr}")
        set_random_seeds(seed=2020, device=DEVICE)
        model = ModelClass().to(DEVICE)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 只需记录每个 step 的 loss，不保存模型
        _, _, _, step_losses = train_one_model(
            model, optimizer, criterion, train_loader, val_loader,
            epochs=epochs, scheduler=None, best_model_filename=None
        )
        all_step_losses.append(step_losses)

    # 将 step_losses 转为 NumPy 数组：shape = [num_lrs, epochs, batches_per_epoch]
    num_lrs = len(lr_list)
    batches_per_epoch = len(train_loader)
    loss_tensor = np.zeros((num_lrs, epochs, batches_per_epoch), dtype=float)

    for i, step_losses in enumerate(all_step_losses):
        arr = np.array(step_losses)
        assert arr.shape == (epochs, batches_per_epoch), \
            f"Expected shape {(epochs, batches_per_epoch)}, got {arr.shape}"
        loss_tensor[i] = arr

    total_steps = epochs * batches_per_epoch
    min_curve = []
    max_curve = []

    # 依次遍历 (epoch, batch_idx)，跨 lr 取 min/max
    for epi in range(epochs):
        for step in range(batches_per_epoch):
            vals = loss_tensor[:, epi, step]
            min_curve.append(vals.min())
            max_curve.append(vals.max())

    # 绘制 Loss 稳定性区间图
    print(f"--> Plotting loss stability for {model_name}")
    plt.figure(figsize=(10, 5))
    steps = np.arange(total_steps)
    plt.plot(steps, min_curve, label="Min Loss (across LRs)")
    plt.plot(steps, max_curve, label="Max Loss (across LRs)")
    plt.fill_between(steps, min_curve, max_curve, alpha=0.3)
    plt.xlabel("Step Index (epoch * batch_idx)")
    plt.ylabel("Loss")
    plt.title(f"Loss Stability: {model_name}")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(FIGURES_PATH, f"loss_stability_{model_name}.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved: {fname}\n")

    return min_curve, max_curve


def plot_bn_vs_no_bn_loss_stability(curves_no_bn: tuple, curves_bn: tuple) -> None:
    """
    在同一张图中对比 VGG_A（无 BN）与 VGG_A_BatchNorm（有 BN）的 Loss 波动区间。
    输入：
      curves_no_bn = (min_curve_no_bn, max_curve_no_bn)
      curves_bn    = (min_curve_bn,    max_curve_bn)
    """
    print("--> Plotting combined loss stability (BN vs No BN)")
    min_no_bn, max_no_bn = curves_no_bn
    min_bn, max_bn = curves_bn

    total_steps = len(min_no_bn)
    steps = np.arange(total_steps)

    plt.figure(figsize=(10, 6))
    # 无 BN 区间（橙色）
    plt.plot(steps, min_no_bn, color="orange", label="VGG_A (no BN) - Min")
    plt.plot(steps, max_no_bn, color="orange", linestyle="--", label="VGG_A (no BN) - Max")
    plt.fill_between(steps, min_no_bn, max_no_bn, color="orange", alpha=0.2)

    # 有 BN 区间（蓝色）
    plt.plot(steps, min_bn, color="blue", label="VGG_A_BN - Min")
    plt.plot(steps, max_bn, color="blue", linestyle="--", label="VGG_A_BN - Max")
    plt.fill_between(steps, min_bn, max_bn, color="blue", alpha=0.2)

    plt.xlabel("Step Index (epoch * batch_idx)")
    plt.ylabel("Loss")
    plt.title("Compare Loss Stability: No BN vs With BN")
    plt.legend()
    plt.grid(True)
    fname = os.path.join(FIGURES_PATH, "loss_stability_bn_vs_no_bn.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved: {fname}\n")


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    set_random_seeds(seed=2020, device=DEVICE)

    # ----------------------------
    # 1. 加载 CIFAR-10 数据集
    # ----------------------------
    print("Loading CIFAR-10 dataset ...")
    train_loader = get_cifar_loader(train=True)    # 使用原始签名
    val_loader = get_cifar_loader(train=False)
    print(f"Dataset loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # （可选）保存一张训练集中样本图片以确认数据正常
    for X_sample, y_sample in train_loader:
        img = X_sample[0].cpu().numpy().transpose(1, 2, 0)
        plt.figure(figsize=(2, 2))
        plt.imshow(img * 0.5 + 0.5)
        plt.title(f"Label: {y_sample[0].item()}")
        plt.axis("off")
        savepath = os.path.join(FIGURES_PATH, "sample_cifar10.png")
        plt.savefig(savepath)
        plt.close()
        print(f"Saved sample image: {savepath}\n")
        break

    # ----------------------------
    # 2. 对比训练：VGG_A vs VGG_A_BN
    # ----------------------------
    print("==== Stage 1: Training and comparing VGG_A vs VGG_A_BatchNorm ====\n")

    # -- 训练 VGG_A（无 BN） --
    print("--> Stage 1.1: VGG_A (no BN)")
    set_random_seeds(seed=2020, device=DEVICE)
    model_no_bn = VGG_A().to(DEVICE)
    optimizer_no_bn = Adam(model_no_bn.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss_A, accA_tr, accA_val, _ = train_one_model(
        model_no_bn,
        optimizer_no_bn,
        criterion,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        scheduler=None,
        best_model_filename="best_vgg_A.pth"
    )

    # -- 训练 VGG_A_BatchNorm（有 BN） --
    print("--> Stage 1.2: VGG_A_BatchNorm")
    set_random_seeds(seed=2020, device=DEVICE)
    model_bn = VGG_A_BatchNorm().to(DEVICE)
    optimizer_bn = Adam(model_bn.parameters(), lr=1e-3)
    loss_BN, accBN_tr, accBN_val, _ = train_one_model(
        model_bn,
        optimizer_bn,
        criterion,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        scheduler=None,
        best_model_filename="best_vgg_A_BatchNorm.pth"
    )

    # 绘制训练曲线对比图
    plot_train_curves(
        EPOCHS,
        loss_A, accA_tr, accA_val,
        loss_BN, accBN_tr, accBN_val
    )

    # ----------------------------
    # 3. Loss 稳定性实验（无 BN）
    # ----------------------------
    print("==== Stage 2: Loss Stability Experiment ====\n")
    print("--> Stage 2.1: VGG_A (no BN)")
    curves_no_bn = run_loss_stability_experiment(
        VGG_A,
        LR_LIST,
        train_loader,
        val_loader,
        epochs=EPOCHS
    )

    # ----------------------------
    # 4. Loss 稳定性实验（带 BN）
    # ----------------------------
    print("--> Stage 2.2: VGG_A_BatchNorm (with BN)")
    curves_bn = run_loss_stability_experiment(
        VGG_A_BatchNorm,
        LR_LIST,
        train_loader,
        val_loader,
        epochs=EPOCHS
    )

    # ----------------------------
    # 5. 对比无 BN 与 带 BN 的 Loss 波动区间
    # ----------------------------
    plot_bn_vs_no_bn_loss_stability(curves_no_bn, curves_bn)

    print("All done. Check figures under:", FIGURES_PATH)
