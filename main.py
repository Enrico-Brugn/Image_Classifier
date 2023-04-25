from data_loader import WireDataset
import torch
import math
from torch.utils.data import DataLoader
from ML import Net

dataset = WireDataset("Input_Data.csv")


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



generator = torch.Generator().manual_seed(4)

train_fraction = math.floor(len(dataset)*0.7)
test_fraction = len(dataset) - train_fraction

test_data_ind, train_data_ind = torch.utils.data.random_split(dataset, [test_fraction, train_fraction], generator = generator)

dl_train = DataLoader(train_data_ind, batch_size=32, shuffle=True)
dl_test = DataLoader(test_data_ind, batch_size=32, shuffle=True)


net = Net()



for i in dl_train:
    print(i)
    x = net(i[0])
    print(x)
    print(x.shape)

    break