import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

torch.manual_seed(42)
np.random.seed(42)


class TransformedDataset(Dataset):
    def __init__(self, file_paths = '/mnt/d/stock_v2510/data/original_stock_history_data.csv',days = 32,label_days=1):
        self.file_paths = file_paths
        df = pd.read_csv(file_paths)
        print(df.columns)
        print(df.shape)

        columns = ['open', 'high', 'low', 'close', 'vol', 'amount']
        for col in columns:
            if df.loc[:,col].min() <=0:
                df.loc[:,col] = df.loc[:,col] - 2*df.loc[:,col].min()

        datas = df.loc[:,columns].values

        log_return = np.diff(np.log(datas + 1), axis=0)  # shape: (n-1, m)

        # 检查 nan/inf
        if not np.isfinite(log_return).all():
            print("Warning: log_return contains nan or inf!")
            # 可选：替换 inf 为大数，nan 为 0
            log_return = np.nan_to_num(log_return, nan=0.0, posinf=1e6, neginf=-1e6)

        self.datas = log_return

        self.days = days
        self.label_days = label_days

    def __len__(self):
        return max(0, self.datas.shape[0] - self.days - self.label_days + 1)

    def __getitem__(self, idx):
        # 输入: [idx, idx + days)
        x = self.datas[idx: idx + self.days]  # shape: (days, m)

        # 标签: [idx + days, idx + days + label_days)
        y = self.datas[idx + self.days: idx + self.days + self.label_days]  # shape: (label_days, m)

        # 通常我们只预测某个特定特征（如 close 的 log return）
        # 假设 close 是第 2 列（open, high, close, low, ... → index=2）
        y = y[:, 2]  # 只取 close 的 return，shape: (label_days,)

        # 如果 label_days == 1，可 squeeze 成标量
        # if self.label_days == 1:
        #     y = y.item()  # 或 y = y[0]
        y = np.array([y])  # 转为 shape (1,)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def plot_line(x,y):
    # 将 y 放到 x 的末尾
    y = np.array(y).flatten()
    line = np.array(x).flatten()

    print(line.shape,y.shape)
    line = np.concatenate([line,y])

    plt.plot(range(line.shape[0]),line)
    plt.show()


if __name__ == '__main__':
    dataset = TransformedDataset()
    print(len(dataset))
    
    item = dataset[3]
    plot_line(item[0].data.numpy()[:,2],item[1])
    item = dataset[30]
    plot_line(item[0].data.numpy()[:, 2], item[1])
    item = dataset[60]
    plot_line(item[0].data.numpy()[:, 2], item[1])