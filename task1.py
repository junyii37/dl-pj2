import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime

from mynn import Resnet18plus, train_model, test_model, set_random_seed


def get_train_dataloaders(data_dir: str, batch_size: int, num_workers: int, augmentation: bool):
    # 数据预处理
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 基本裁剪
            transforms.RandomHorizontalFlip(),  # 水平翻转
            AutoAugment(AutoAugmentPolicy.CIFAR10),  # AutoAugment
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            ),
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                inplace=True
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])

    # 加载原始训练集
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)

    # 拆分原始训练集
    train_len = int(0.9 * len(full_train))
    val_len = len(full_train) - train_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    # 构建 dataloader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def get_test_dataloader(data_dir: str, batch_size: int, num_workers: int):
    # 数据预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    # 加载测试数据集
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # 构建 dataloader
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_loader


def train(model):
    # 环境配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_workers = 0 if os.name == 'nt' else 4

    # 加载数据
    train_loader, val_loader = get_train_dataloaders(data_dir='./data', batch_size=128, num_workers=num_workers, augmentation=True)

    # 模型
    model = model

    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 训练轮数
    num_epochs = 150
    # 学习率调度
    scheduler = None
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    # 容忍度
    patience = 15
    # 最小精度提升
    delta = 0.0001

    # 训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        delta=delta,
        scheduler=scheduler,
    )


def test(block_type, deeper_classifier, best_model_path=None):
    # 环境配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    num_workers = 0 if os.name == 'nt' else 4

    # 加载数据
    test_loader = get_test_dataloader(data_dir='./data', batch_size=128, num_workers=num_workers)

    # 初始化模型和损失函数
    # 模型
    model = Resnet18plus(block_type=block_type, deeper_classifier=deeper_classifier)

    criterion = nn.CrossEntropyLoss()

    # 加载最佳模型
    if best_model_path is None:
        current_year = str(datetime.now().year)  # 当前年份
        all_saved = glob.glob("results/*/best_model.pth")
        saved = [
            path for path in all_saved
            if os.path.basename(os.path.dirname(path)).startswith(current_year)
        ]
        if not saved:
            raise FileNotFoundError("No model saved, please run train.py first.")
        best_model_path = sorted(saved)[-1]

    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    # 测试
    test_model(model, test_loader, criterion, device)

    return model


if __name__ == '__main__':
    set_random_seed()

    # model = Resnet18plus(
    #     block_type='preact_se',
    #     num_classes=10,
    #     drop_prob_max=0.2,
    #     p_dropout=0.35,
    #     deeper_classifier=False
    # )
    # train(model)

    model = test(block_type='preact_se', deeper_classifier=False, best_model_path="results/ResNet18Plus+SGD+OneCycleLR/best_model.pth")
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # 导出为 ONNX
    # dummy_input = torch.randn(1, 3, 32, 32).to(device)
    # torch.onnx.export(model, dummy_input, "resnet18plus.onnx",
    #                   input_names=["input"], output_names=["output"],
    #                   opset_version=12)

