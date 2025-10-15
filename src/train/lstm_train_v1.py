import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# ----------------------------
# 1. 模拟金融价格数据（或替换为真实数据）
# ----------------------------
np.random.seed(42)
T = 1000  # 总时间步
# 模拟几何布朗运动（GBM）：金融价格典型模型
dt = 1 / 252  # 年化（252交易日）
mu = 0.0002  # 日漂移
sigma = 0.02  # 日波动率

# 生成价格
log_price = np.zeros(T)
log_price[0] = np.log(100)  # 初始价格=100
for t in range(1, T):
    log_price[t] = log_price[t - 1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn()

price = np.exp(log_price)  # (T,)

# ----------------------------
# 2. 计算对数收益率（log return）
# ----------------------------
log_returns = np.diff(log_price)  # shape: (T-1,)


# 注意：log_returns[t] 对应从 t 到 t+1 的收益率

# ----------------------------
# 3. 构造时序样本（滑动窗口，无穿越）
# ----------------------------
def create_sequences(data, seq_len, pred_horizon=1):
    """
    data: 1D array of log returns (length = L)
    返回:
        X: (num_samples, seq_len, 1)
        y: (num_samples,)  # 预测未来 pred_horizon 步的 log return
    """
    X, y = [], []
    # 确保 y 不越界: i + seq_len + pred_horizon - 1 < len(data)
    for i in range(len(data) - seq_len - pred_horizon + 1):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len: i + seq_len + pred_horizon])
    X = np.array(X)[..., np.newaxis]  # (N, seq_len, 1)
    y = np.array(y).squeeze()  # (N,) if pred_horizon=1
    return X, y


seq_len = 20  # 用过去20天预测下1天
pred_horizon = 1  # 预测未来1步

# 注意：log_returns 长度 = T-1
X, y = create_sequences(log_returns, seq_len, pred_horizon)

# 划分训练/验证集（时间顺序！）
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


# ----------------------------
# 4. 自定义 Dataset
# ----------------------------
class ReturnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(ReturnDataset(X_train, y_train), batch_size=32, shuffle=False)
val_loader = DataLoader(ReturnDataset(X_val, y_val), batch_size=32, shuffle=False)


# ----------------------------
# 5. LSTM 模型（预测 log return）
# ----------------------------
class LSTMReturnPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, L, H)
        out = self.fc(out[:, -1, :])  # (B, 1)
        return out.squeeze(-1)  # (B,)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMReturnPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 6. 训练循环
# ----------------------------
epochs = 50
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_val_loss += loss.item()

    train_losses.append(total_train_loss / len(train_loader))
    val_losses.append(total_val_loss / len(val_loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

# ----------------------------
# 7. 预测并还原为绝对价格
# ----------------------------
# 取验证集最后一个窗口作为起点
last_window = X_val[-1:]  # shape: (1, seq_len, 1)
last_price = price[split + seq_len + len(y_train)]  # 对应预测起点的真实价格

model.eval()
with torch.no_grad():
    pred_log_return = model(torch.tensor(last_window, dtype=torch.float32).to(device)).item()

# 还原预测价格
pred_price = last_price * np.exp(pred_log_return)
true_price = price[split + seq_len + len(y_train) + 1]  # 真实下一期价格

print(f"\n预测 log return: {pred_log_return:.6f}")
print(f"基于价格 {last_price:.2f}，预测下一期价格: {pred_price:.2f}")
print(f"真实价格: {true_price:.2f}")
print(f"预测误差: {abs(pred_price - true_price):.2f}")

# ----------------------------
# 8. （可选）可视化训练损失
# ----------------------------
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()