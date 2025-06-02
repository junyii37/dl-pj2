import torch
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR, CosineAnnealingWarmRestarts
import random
import numpy as np


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5, delta=0.001, scheduler=None, device='cpu'):
    model = model.to(device)

    # 建立训练日志
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.join("results", time_str)
    os.makedirs(folder_name, exist_ok=True)
    writer = SummaryWriter(log_dir=folder_name)
    model_path = os.path.join(folder_name, "best_model.pth")

    # 实用变量
    best_acc, no_improve_epochs = float('-inf'), 0
    batch_index = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(iterable=train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}: ", unit="batch",
                    file=sys.stdout, colour="red")

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # 更新模型参数
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            # 记录中间结果
            running_loss += loss.item()
            _, preds = logits.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            # 可视化
            acc = preds.eq(targets).sum().item() / targets.size(0)
            batch_index += 1
            writer.add_scalar("Running Loss", loss.item(), batch_index)
            writer.add_scalar("Running Accuracy", acc, batch_index)

            # 更新进度条
            loop.set_postfix({
                'batch_loss': loss.item(),
                'batch_accuracy': acc,
            })

            # Batch-wise 调度: OneCycleLR, CyclicLR, CosineAnnealingWarmRestarts
            if scheduler is not None and isinstance(scheduler, (OneCycleLR, CyclicLR, CosineAnnealingWarmRestarts)):
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("LR", current_lr, batch_index)


        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = val_model(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Accuracy: {val_acc:.4f}")

        # 可视化
        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={
                "Train": train_loss,
                "Val": val_loss,
            },
            global_step=epoch,
        )
        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={
                "Train": train_acc,
                "Val": val_acc,
            },
            global_step=epoch,
        )

        # Epoch-wise 调度
        if scheduler is not None and not isinstance(scheduler, (OneCycleLR, CyclicLR, CosineAnnealingWarmRestarts)):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)

        if val_acc > best_acc + delta:  # 精度提升超过最小要求
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"Best model so far saved at {model_path}\n")
        else:  # 精度提升未超过最小要求
            no_improve_epochs += 1
            if no_improve_epochs >= patience:  # 满容忍度
                print(f"Early stopping triggered after {epoch+1} epochs.\n")
                break
            print()

    if no_improve_epochs < patience:
        print(f"Training finished after {epoch+1} epochs.")
    print(f"Best training accuracy on validation set: {best_acc:.4f}")
    print(f"Best model saved at {model_path}\n")

    writer.close()



def val_model(model, loader, criterion, device='cpu'):
    model.to(device)
    model.eval()  # 设置为测试模式

    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item()
            _, preds = logits.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    loss = running_loss / len(loader)
    acc = correct / total
    return  loss, acc


def test_model(model, loader, criterion, device='cpu'):
    loss, acc = val_model(model, loader, criterion, device)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)