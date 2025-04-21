import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm as WN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 250

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             download=True,
                                             transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


class CNN(nn.Module):
    def __init__(self, hidden_size, output_size, optim, norm):
        super().__init__()
        filter_h, filter_w, filter_c, filter_n = 5, 5, 1, 30

        self.conv1 = nn.Conv2d(filter_c, filter_n, (filter_w, filter_h))

        self.norm1 = None
        self.norm2 = None

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12 * 12 * filter_n, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        if norm is LayerNorm or norm is nn.LayerNorm:
            self.norm1 = norm([filter_n, 24, 24])
            self.norm2 = norm([hidden_size])
        elif norm is BatchNorm or norm is nn.BatchNorm2d:
            self.norm1 = norm(filter_n)
            self.norm2 = norm(hidden_size, 2)
        elif norm is WeightNorm or norm is WN:
            self.fc1 = norm(self.fc1)
            # self.conv1 = norm(self.conv1)
        if norm is nn.BatchNorm2d:
            self.norm2 = nn.BatchNorm1d(hidden_size)

        self.optim = optim(self.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = x.view(-1, 12 * 12 * 30)
        x = self.fc1(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def train_step(self, loader):
        self.train()

        for data in loader:
            inputs, labels = data

            self.optim.zero_grad()
            outputs = self.forward(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optim.step()

        return loss.item()

    def test_model(self, loader):
        self.eval()
        total = 0
        correct = 0
        for data in loader:
            images, labels = data

            outputs = self.forward(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total


class BatchNorm(nn.Module):

    def __init__(self, n=30, n_dim=4, epsilon=1e-5):
        super().__init__()

        self.n_dim = n_dim
        shape = (1, n, 1, 1)
        if self.n_dim == 2:
            shape = (1, n)
        self.epsilon = epsilon
        self.mu = None
        self.var = None
        self.gamma = torch.nn.Parameter(torch.ones(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        if self.training:
            if self.n_dim == 2:
                self.mu = x.mean(axis=0)
                self.var = x.var(axis=0, unbiased=True)
            else:
                self.mu = x.mean((0, 2, 3), keepdim=True)
                self.var = x.var((0, 2, 3), keepdim=True, unbiased=True)

        x_hat = (x - self.mu) \
            / torch.sqrt(self.var + self.epsilon)
        return x_hat * self.gamma + \
            self.beta


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape=[30, 24, 24],  epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.mu = None
        self.var = None
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.dim = [-(dim + 1) for dim, _ in enumerate(normalized_shape)]

    def forward(self, x):
        if self.training:
            self.mu = x.mean(self.dim, keepdim=True)
            self.var = x.var(self.dim, keepdim=True)

        x_hat = (x - self.mu) \
            / torch.sqrt(self.var + self.epsilon)
        return x_hat * self.gamma + \
            self.beta


class WeightNorm(nn.Module):
    def __init__(self, layer: torch.nn.Module, name='weight'):
        super().__init__()
        self.layer = layer

        self.v = torch.nn.Parameter(torch.normal(mean=0.,
                                                 std=0.05,
                                                 size=(layer.weight.shape)
                                                 ))
        self.g = torch.nn.Parameter(torch.normal(mean=0.0,
                                                 std=0.05,
                                                 size=(1,)
                                                 ))

    def forward(self, x):
        weight = (self.g / torch.norm(self.v.T, dim=-1)) * self.v
        outputs = torch.matmul(x, weight.T) + self.layer.bias
        return outputs


norms = {"BatchNorm (me)": BatchNorm,
         "BatchNorm (torch)": nn.BatchNorm2d,
         "LayerNorm (me)": LayerNorm,
         "LayerNorm (torch)": nn.LayerNorm,
         "WeightNorm (me)": WeightNorm,
         "WeightNorm (torch)": WN,
         "No Norm": None}


num_classes = 10
hidden_size = 50
n_epochs = 5


def run_experiment(norms, output):
    results = pd.DataFrame(columns=["Test Acc", "Train Loss", "Norm"],
                           dtype=object)

    for key in norms:
        train_losses = np.zeros((n_epochs))
        torch.manual_seed(99111100121)
        model = CNN(hidden_size, num_classes, optim.Adam, norms[key])
        print(f"Training with norm: {key}")
        for epoch in range(n_epochs):
            train_loss = model.train_step(trainloader)
            print(f"Epoch {epoch}: {train_loss}")
            train_losses[epoch] = train_loss
        test_acc = model.test_model(testloader)
        print(f"Test Acc {test_acc}")

        new_r = pd.DataFrame({"Test Acc": test_acc,
                              "Train Loss": [train_losses],
                              "Norm": key})

        results = pd.concat([results, new_r], ignore_index=True)

    return results


results = run_experiment(norms, None)


colors = ["red", "blue", "green", "black", "yellow", "orange", "pink"]
plt.bar(results["Norm"].to_list(), results["Test Acc"].to_numpy(),
        color=colors, log=True)
plt.ylabel("Log of Test Accuracy")
plt.xlabel("Type of Normalization")

plt.show()

for loss, norm, color in zip(results["Train Loss"], results["Norm"], colors):
    plt.plot(loss, label=norm, color=color, linewidth=2.5)

plt.xticks([0, 1, 2, 3, 4])
plt.xlabel("Training Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.show()

plt.bar("BatchNorm",
        -np.diff(results[results["Norm"].str.contains("BatchNorm")]
                 ["Test Acc"].to_numpy()), color="red")
plt.bar("LayerNorm",
        -np.diff(results[results["Norm"].str.contains("LayerNorm")]
                 ["Test Acc"].to_numpy()), color="green")
plt.bar("WeightNorm",
        -np.diff(results[results["Norm"].str.contains("WeightNorm")]
                 ["Test Acc"].to_numpy()), color="blue")
plt.show()
