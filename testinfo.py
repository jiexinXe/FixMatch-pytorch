import torch
from torchvision import models, transforms, datasets
import os

from dataset.cifar import info_prescreen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载一个预训练模型，例如 ResNet18（用于 CIFAR10 数据需调整最后一层）
model = models.resnet18(pretrained=True)
# 修改最后一层以适应 CIFAR10（10 类）
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.to(device)

# 设置数据集路径
data_root = "./datasets/cifar10"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])
cifar10_train = datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True)

# 调用预筛选函数
selected_idxs = info_prescreen(cifar10_train, model, device, keep_ratio=0.5, min_keep_ratio=0.2, batch_size=128)

# 输出每个类别的样本数
import numpy as np

# 假设 cifar10_train 已经加载好了
print("原始数据总数：", len(cifar10_train))

# 统计各类别样本数（假设 cifar10_train.targets 为标签列表或数组）
original_counts = {}
for cls in range(10):
    original_counts[cls] = np.sum(np.array(cifar10_train.targets) == cls)
for cls in sorted(original_counts.keys()):
    print(f"类别 {cls} 原始样本数：{original_counts[cls]}")



targets = np.array(cifar10_train.targets)
for cls in range(10):
    count = np.sum(targets[selected_idxs] == cls)
    print(f"类别 {cls} 筛选后样本数：{count}")

print("InfoPrescreen: 原始数据数 =", len(cifar10_train), "筛选后数据数 =", len(selected_idxs))