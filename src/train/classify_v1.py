import torch
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (used by PyTorch):", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(0))


torch.manual_seed(42)
np.random.seed(42)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# ----------------------------
# 1. 生成模拟数据（你可以替换成自己的数据）
# ----------------------------
np.random.seed(42)
torch.manual_seed(42)

N = 10000  # 样本数
D = 10  # 特征维度
X = np.random.randn(N, D).astype(np.float32)  # 连续浮点特征

# 生成连续标签：-10.0 ~ 10.0
y_cont = np.random.uniform(-10.0, 10.0, size=(N,)).astype(np.float32)

# ----------------------------
# 2. 分箱（Binning）：将连续标签转为类别
# ----------------------------
n_bins = 20  # 分成 20 个类别（可调）
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
y_binned = discretizer.fit_transform(y_cont.reshape(-1, 1)).astype(np.int64).flatten()

print("分箱标签:",np.unique(y_binned))

print(f"原始标签范围: [{y_cont.min():.2f}, {y_cont.max():.2f}]")
print(f"分箱后类别数: {len(np.unique(y_binned))} (目标: {n_bins})")
print(f"类别分布（前5类）: {np.bincount(y_binned)[:5]}")

# ----------------------------
# 3. 划分训练/测试集
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binned, test_size=0.2, random_state=42, stratify=y_binned
)

# 转为 PyTorch 张量
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ----------------------------
# 4. 定义分类模型
# ----------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)  # 输出 logits
        )

    def forward(self, x):
        return self.net(x)


model = MLPClassifier(input_dim=D, num_classes=n_bins)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ----------------------------
# 5. 训练循环
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 500
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            pred = logits.argmax(dim=1)
            test_correct += (pred == y_batch).sum().item()
            test_total += y_batch.size(0)
    test_acc = test_correct / test_total

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

# ----------------------------
# 6. 评估 & 可视化（可选）
# ----------------------------
print("\n✅ 训练完成！")

# 可选：查看预测 vs 真实分箱
model.eval()
with torch.no_grad():
    sample_x = X_test_t[:5].to(device)
    sample_y_bin = y_test_t[:5].cpu().numpy()
    logits = model(sample_x)
    pred_bin = logits.argmax(dim=1).cpu().numpy()

print("\n样本预测（分箱索引）:")
for i in range(5):
    print(f"真实分箱: {sample_y_bin[i]}, 预测分箱: {pred_bin[i]}")

# 如果你想还原为原始浮点范围（近似）：
# 每个 bin 的中心值 ≈ discretizer.bin_edges_[0][bin_idx] 和 [bin_idx+1] 的中点
edges = discretizer.bin_edges_[0]
bin_centers = (edges[:-1] + edges[1:]) / 2

print("\n还原为近似原始值（仅作参考）:")
for i in range(5):
    true_val = bin_centers[sample_y_bin[i]]
    pred_val = bin_centers[pred_bin[i]]
    print(f"真实≈{true_val:.2f}, 预测≈{pred_val:.2f}")


# 修改学习率（保留优化器状态！）
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-2