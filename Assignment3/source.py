import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.functional import one_hot
import pandas as pd

torch.random.manual_seed(1)

train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transforms.ToTensor())

train_dataset.data = train_dataset.data / 255.
test_dataset.data = test_dataset.data / 255.


train_dataset.data = train_dataset.data.reshape([-1, 784])
test_dataset.data = test_dataset.data.reshape([-1, 784])

test_labels = test_dataset.targets
train_labels = train_dataset.targets

num_tasks = 10
num_classes = 10
epochs_per_run = 20
batch_size = 10000

train_task_images = [train_dataset.data]
test_task_images = [test_dataset.data]

softmax = torch.nn.Softmax(dim=-1)
lsoftmax = torch.nn.LogSoftmax(dim=-1)


for task in range(num_tasks-1):
    perm = torch.randperm(784)
    train_task_images.append(train_dataset.data[:, perm])
    test_task_images.append(test_dataset.data[:, perm])

hwidth = 256


class NewSet(datasets.MNIST):
    def __init__(self, data, targets):
        super().__init__(root='./data', train=True, download=True)

        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        return (self.data[idx], self.targets[idx])


class MLP(torch.nn.Module):

    def __init__(self, depth, dropout_prob, opt):
        super().__init__()

        self.stack = torch.nn.Sequential(
                torch.nn.Linear(28*28, hwidth, bias=True),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_prob),
        )

        for i in range(1, depth-1):
            self.stack.append(torch.nn.Linear(hwidth, hwidth, bias=True))
            self.stack.append(torch.nn.ReLU())
            self.stack.append(torch.nn.Dropout(dropout_prob))

        self.stack.append(torch.nn.Linear(hwidth, 10, bias=True))

        self.optimizer = opt(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.stack(x)

    def train_step(self, x, labels, loss_type):
        model.train(True)
        logits = self(x)
        loss = self.compute_loss(logits, labels, loss_type)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, loss

    def compute_loss(self, logits, labels, loss_type):
        loss_fns = {
                "NLL": lambda logits, y: torch.nn.CrossEntropyLoss()(
                 lsoftmax(logits), y),
                "L2": lambda logits, y: torch.nn.MSELoss()(
                 softmax(logits), y),
                "L1": lambda logits, y: torch.nn.L1Loss()(
                 softmax(logits), y),
                "L1+L2": lambda logits, y:
                torch.nn.MSELoss()(softmax(logits), y)
                + torch.nn.L1Loss()(softmax(logits), y)
                    }
        y = one_hot(labels, num_classes).float()
        return loss_fns[loss_type](logits, y)


def train_model(task_num, model, epochs, loss_t):

    for epoch in range(epochs):
        new = NewSet(train_task_images[task_num], train_labels)
        loader = DataLoader(new, batch_size=batch_size, shuffle=True)
        for batch in loader:
            inputs, targets = batch
            logits, loss = model.train_step(inputs, targets, loss_t)
        print(f'Epoch: {epoch}, Loss: {loss}')
    return model


def test_model(model, inputs, targets):
    model.eval()
    correct = torch.sum(softmax(model(inputs)).argmax(dim=1) == targets)
    return correct / targets.shape[0]


def run_sequential_training(model, loss_t):
    R = torch.zeros((num_tasks, num_tasks))

    for task_idx in range(num_tasks):

        epochs = epochs_per_run

        if task_idx == 0:
            epochs = 50
        print(f"Task: {task_idx}")
        model = train_model(task_idx, model, epochs, loss_t)

        for i in range(task_idx+1):
            accuracy = test_model(model, test_task_images[i], test_labels)
            R[task_idx, i] = accuracy

    return R


def calc_acc(R):
    return torch.sum(R[num_tasks-1, :]) / num_tasks


def calc_bwt(R):
    bwt = 0
    for i in range(num_tasks - 1):
        bwt += (R[num_tasks - 1, i] - R[i, i])

    return bwt / (num_tasks - 1)


def log_results(acc, bwt, config):
    loss_t, optim, depth, rate = config
    new_result = pd.DataFrame({"Loss": [loss_t],
                               "Optimizer": [optim],
                               "Depth": [depth],
                               "Dropout": [rate],
                               "Acc": [acc],
                               "BWT": [bwt]})
    return new_result


loss_types = ["NLL", "L1", "L2", "L1+L2"]
optimizers = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam,
              "RMSProp": torch.optim.RMSprop}
depths = [2, 3, 4]
dropout_rates = [0.2, 0.4, 0.6]

results = pd.DataFrame(columns=["Loss", "Optimizer", "Depth",
                                "Dropout", "Acc", "BWT"])

for loss_t in loss_types:
    for optim in optimizers.keys():
        for depth in depths:
            for rate in dropout_rates:
                model = MLP(depth, rate, optimizers[optim])

                R = run_sequential_training(model, loss_t)

                acc = calc_acc(R).item()
                bwt = calc_bwt(R).item()

                new_r = log_results(acc, bwt, (loss_t, optim, depth, rate))
                results = pd.concat([results, new_r], ignore_index=True)
