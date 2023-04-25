from data_loader import WireDataset
import torch
import math
from torch.utils.data import DataLoader


dataset = WireDataset("Input_Data.csv")
generator = torch.Generator().manual_seed(4)

train_fraction = math.floor(len(dataset)*0.7)
test_fraction = len(dataset) - train_fraction

test_data_ind, train_data_ind = torch.util.data.random_split(dataset, [test_fraction, train_fraction], generator = generator)

dl_train = DataLoader(train_data_ind, batch_size=32, shuffle=True)
dl_test = DataLoader(test_data_ind, batch_size=32, shuffle=True)