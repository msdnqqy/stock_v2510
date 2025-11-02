import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/mnt/d/projects/stock_v2510', '/mnt/d/projects/stock_v2510/src'])

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from data_loader.stock_dataset_v2 import TransformedDataset

# ----------------------------
# 1. 模拟金融价格数据（或替换为真实数据）
# ----------------------------
np.random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
import torch
from torch.utils.data import DataLoader, Subset


def get_datasets(paths=['./data/stock_pre/SH#600031.csv']):
    result=[]
    for path in paths:
        # 1. 创建完整数据集
        full_dataset = TransformedDataset(
            file_paths = path,
            days=32,
            label_days=1
        )

        # 2. 计算划分点（按时间顺序！）
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        print(f"Total samples: {total_size}")
        print(f"Train samples: {train_size}")
        print(f"Val samples: {val_size}")

        # 3. 创建训练集和验证集（使用索引切片）
        train_dataset = Subset(full_dataset, indices=range(0, train_size))
        val_dataset = Subset(full_dataset, indices=range(train_size, total_size))

        # 4. 创建 DataLoader（注意：时序数据通常不 shuffle 训练集！）
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 5. 验证划分是否正确
        print("First train sample index:", train_dataset.indices[0])
        print("Last train sample index:", train_dataset.indices[-1])
        print("First val sample index:", val_dataset.indices[0])
        print("Last val sample index:", val_dataset.indices[-1])

        result.append((train_loader, val_loader))

    return result


data_loaders = get_datasets(paths = ['./data/stock_pre/SH#600031.csv'
    ,'./data/stock_pre/SH#600036.csv'

    # run（1）
    ,'./data/stock_pre/SH#600048.csv'
    ,'./data/stock_pre/SH#600050.csv'
    ,'./data/stock_pre/SH#600111.csv'

    # run(2)
    ,'./data/stock_pre/SH#600276.csv'
    ,'./data/stock_pre/SH#600309.csv'
    ,'./data/stock_pre/SH#600406.csv'
    ,'./data/stock_pre/SH#600415.csv'
    ,'./data/stock_pre/SH#600426.csv'
    ,'./data/stock_pre/SH#600436.csv'
    ,'./data/stock_pre/SH#600519.csv'

    # ,'./data/stock_pre/SH#600585.csv'
    # ,'./data/stock_pre/SH#600660.csv'
    # ,'./data/stock_pre/SH#600690.csv'
    # ,'./data/stock_pre/SH#600809.csv'
    # ,'./data/stock_pre/SH#600887.csv'
    # ,'./data/stock_pre/SH#600893.csv'
    # ,'./data/stock_pre/SH#600900.csv'
    # ,'./data/stock_pre/SH#600919.csv'
    # ,'./data/stock_pre/SH#600941.csv'
    # ,'./data/stock_pre/SH#601012.csv'
    # ,'./data/stock_pre/SH#601088.csv'
    # ,'./data/stock_pre/SH#601138.csv'
    # ,'./data/stock_pre/SH#601166.csv'
                                     ])
# ----------------------------
# 5. LSTM 模型（预测 log return）
# ----------------------------
class LSTMReturnPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=1024, num_layers=3, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        self.net = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_size)  # 输出 logits
        )


    def forward(self, x):
        out, _ = self.lstm(x)  # (B, L, H)
        out = self.net(out[:, -1, :])  # (B, 1)

        return out
        # return out.squeeze(-1)  # (B,)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMReturnPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 6. 训练循环
# ----------------------------
epochs = 50
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for data_loader in data_loaders:
        for X_batch, y_batch in data_loader[0]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 👈 关键！
            optimizer.step()
            total_train_loss += loss.item()

    # 验证
    model.eval()
    total_val_loss = 0
    for data_loader in data_loaders:
        with torch.no_grad():
            for X_batch, y_batch in data_loader[1]:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_val_loss += loss.item()
        break

    # print(total_train_loss,len(train_loader),total_val_loss,len(val_loader))
    train_losses.append(total_train_loss / np.array([len(data_loader[0]) for data_loader in data_loaders]).sum())
    val_losses.append(total_val_loss /  np.array([len(data_loader[1]) for data_loader in data_loaders]).sum())

    len_val = 0
    for data_loader in data_loaders:
        len_val+=len(data_loader[1])
        break
        
    val_losses.append(total_val_loss /  len_val)

    if True or (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

# ----------------------------
# 7. 预测并还原为绝对价格
# ----------------------------
# 取验证集最后一个窗口作为起点
# last_window = X_val[-1:]  # shape: (1, seq_len, 1)
# last_price = price[split + seq_len + len(y_train)]  # 对应预测起点的真实价格
#
# model.eval()
# with torch.no_grad():
#     pred_log_return = model(torch.tensor(last_window, dtype=torch.float32).to(device)).item()
#
# # 还原预测价格
# pred_price = last_price * np.exp(pred_log_return)
# true_price = price[split + seq_len + len(y_train) + 1]  # 真实下一期价格
#
# print(f"\n预测 log return: {pred_log_return:.6f}")
# print(f"基于价格 {last_price:.2f}，预测下一期价格: {pred_price:.2f}")
# print(f"真实价格: {true_price:.2f}")
# print(f"预测误差: {abs(pred_price - true_price):.2f}")
#
# # ----------------------------
# # 8. （可选）可视化训练损失
# # ----------------------------
# plt.figure(figsize=(10, 4))
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Val Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')
# plt.legend()
# plt.grid(True)
# plt.show()