import torch
import torch.nn as nn
from mynn.utils import set_random_seed


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.pool(x).squeeze(-1).squeeze(-1)
        scale = self.fc(y).view(n, c, 1, 1)
        return x * scale


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.drop_path = DropPath(self.drop_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out = self.drop_path(out)
        out += identity
        out = self.relu(out)

        return out


class PreActBlockSE(nn.Module):
    expansion = 1  # 对应 PreActBottleneckSE 中的属性，提高代码可扩展性

    def __init__(self, in_ch, out_ch, stride=1, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

        # Pre-activation + Dropout
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act1 = nn.ReLU()

        # Convolutions
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        # SE module
        self.se = SEBlock(out_ch)

        # Shortcut
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

        # DropPath
        self.drop_path = DropPath(self.drop_prob)

    def forward(self, x):
        identity = x

        # Pre-activation
        out = self.bn1(x)
        out = self.act1(out)

        if self.downsample:
            identity = self.downsample(out)

        # Two convolutional layers
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        # SE module
        out = self.se(out)

        # Stochastic depth
        out = self.drop_path(out)

        # Residual connection
        return out + identity


class Resnet18plus(nn.Module):
    def __init__(self, block_type='preact_se', num_classes=10, drop_prob_max=0.1, p_dropout=0.2, deeper_classifier=False):
        super().__init__()
        self.block_id = 0
        self.block_type = block_type
        if self.block_type == 'basic':
            self.block_cls = BasicBlock
        elif self.block_type == 'preact_se':
            self.block_cls = PreActBlockSE
        else:
            raise TypeError('Unknown block type.')

        # Regularization settings
        self.drop_prob_max = drop_prob_max
        self.p_dropout = p_dropout

        # Initial conv + BN + ReLU + Dropout
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU()

        # Define number of blocks per stage (ResNet-10)
        self.stages = (2, 2, 2, 2)

        # Build residual stages
        self.layer1 = self._make_layer(64, self.stages[0], stride=1)
        self.layer2 = self._make_layer(128, self.stages[1], stride=2)
        self.layer3 = self._make_layer(256, self.stages[2], stride=2)
        self.layer4 = self._make_layer(512, self.stages[3], stride=2)

        # Pooling and dropout
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(p_dropout)

        # Head
        if deeper_classifier:
            self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p_dropout),

                nn.Linear(256, num_classes)
            )
        else:
            self.fc = nn.Linear(self.in_ch, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, out_ch, blocks, stride):
        layers = []
        total_blocks = sum(self.stages)
        for i in range(blocks):
            # Linearly scale drop path probability
            self.block_id += 1
            drop_p = self.drop_prob_max * (self.block_id / total_blocks)
            layers.append(
                self.block_cls(
                    self.in_ch, out_ch,
                    stride=stride if i == 0 else 1,
                    drop_prob=drop_p,
                )
            )
            self.in_ch = out_ch * self.block_cls.expansion
            stride = 1
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = torch.flatten(self.avgpool(x), 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    set_random_seed()
    for blk in ['basic', 'preact_se']:
        print(f"Testing block: {blk}")
        model = Resnet18plus(block_type='basic' if blk == 'basic' else 'preact_se')
        print('-'*20)
        print(model)
        print('-'*20)
        y = model(torch.randn(4, 3, 32, 32))
        print("Output shape:", y.shape)
