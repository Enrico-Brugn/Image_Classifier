from data_loader import WireDataset
import torch
import math
from torch.utils.data import DataLoader
from ML import Net
import torch.nn.functional as F
import matplotlib.pyplot as plt
dataset = WireDataset("Input_Data.csv")


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



generator = torch.Generator().manual_seed(4)

train_fraction = math.floor(len(dataset)*0.7)
test_fraction = len(dataset) - train_fraction

test_data_ind, train_data_ind = torch.utils.data.random_split(dataset, [test_fraction, train_fraction], generator = generator)

dl_train = DataLoader(train_data_ind, batch_size=20, shuffle=True)
dl_test = DataLoader(test_data_ind, batch_size=1, shuffle=True)

net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    net.train()
    for i, data in enumerate(dl_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 30 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
        
            
    # net.eval()
    # running_loss = 0.0
    # for i, data in enumerate(dl_test, 0):
    #     inputs, labels = data
    #     # forward + backward + optimize
    #     outputs = net(inputs)
    #     loss = criterion(outputs, labels)
    #     # print statistics
    #     running_loss += loss.item()
    #     if i % 10 == 9:    # print every 30 mini-batches
    #         print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
    #         running_loss = 0.0
    #         print(outputs)
    #         plt.imshow(inputs.squeeze().detach())
    #         plt.savefig("res1.png")
    #         input()
               

print('Finished Training')