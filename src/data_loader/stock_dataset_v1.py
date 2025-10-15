import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)


class TransformedDataset(Dataset):
    def __init__(self, file_paths = '/mnt/d/stock_v2510/data/original_stock_history_data.csv',days = 32,label_days=1):
        self.file_paths = file_paths
        df = pd.read_csv(file_paths)
        print(df.columns)
        print(df.shape)

        datas = df.loc[:,['open', 'high', 'close', 'low','market_cap','volume']].values
        self.datas = datas

        self.days = days
        self.label_days = label_days

    def __len__(self):
        return self.datas.shape[0] - self.days - self.label_days

    def __getitem__(self, idx):
        scaler = StandardScaler()
        x = scaler.fit_transform(self.datas[idx:idx+self.days][2]).flatten()
        y = scaler.transform(self.datas[idx+self.days:idx+self.days+self.label_days][2]).flatten()
        return x, y


if __name__ == '__main__':
    dataset = TransformedDataset()
    print(len(dataset))
