from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from core.data import BaseDataset
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class DenseModel(nn.Module):
    def __init__(self, input_size:int, num_classes:int, hidden_sizes:tuple):
        assert len(hidden_sizes) > 0
        super().__init__()

        self.inpt = nn.Linear(input_size, hidden_sizes[0])
        self.hidden = []
        for i in range(len(hidden_sizes)):
            self.hidden.append(nn.Linear(hidden_sizes[max(0, i - 1)], hidden_sizes[i]))
        self.out = nn.Linear(hidden_sizes[-1], num_classes)

    def _encode(self, x:Tensor)->Tensor:
        """
        The split bewteen encoding and prediction is important for agents that use latent features from the
        classifier like Coreset
        """
        x = self.inpt(x)
        x = F.relu(x)
        for h_layer in self.hidden:
            x = h_layer(x)
            x = F.relu(x)
        return x

    def forward(self, x:Tensor)->Tensor:
        x = self._encode(x)
        x = self.out(x)
        return x


class ConvolutionalModel(nn.Module):
    def __init__(self, input_size:Tuple[int], num_classes:int, hidden_sizes:Tuple[int]):
        assert len(hidden_sizes) > 0
        assert len(input_size) > 1 and len(input_size) < 4
        if len(input_size) == 2:
            print("found greyscale input. adding a color dimension for compatibility")
            input_size = (1, *input_size)
        super().__init__()

        self.inpt = nn.Conv2d(input_size[0], hidden_sizes[0], kernel_size=3)
        self.hidden = []
        for i in range(len(hidden_sizes)):
            self.hidden.append(nn.Conv2d(hidden_sizes[max(0, i - 1)], hidden_sizes[i], kernel_size=3))
        self.flatten = nn.Flatten()

        test_inpt = torch.zeros((1, *input_size))
        test_out = self._encode(test_inpt)

        self.out = nn.Linear(test_out.shape[-1], num_classes)

    def _encode(self, x:Tensor)->Tensor:
        """
        The split bewteen encoding and prediction is important for agents that use latent features from the
        classifier like Coreset
        """
        x = self.inpt(x)
        x = F.relu(x)
        for h_layer in self.hidden:
            x = h_layer(x)
            x = F.relu(x)
        x = self.flatten(x)
        return x

    def forward(self, x:Tensor)->Tensor:
        x = self._encode(x)
        x = self.out(x)
        return x


def fit_and_evaluate(dataset:BaseDataset,
                     lr:float, weight_decay:float, batch_size:int,
                     hidden_sizes:tuple,
                     disable_progess_bar:bool=False,
                     max_epochs:int=1000):
    loss = nn.CrossEntropyLoss()
    model = dataset.get_classifier(hidden_dims=hidden_sizes)
    model = model.to(dataset.device)
    optimizer = dataset.get_optimizer(model, lr=lr, weight_decay=weight_decay)

    train_dataloader = DataLoader(TensorDataset(dataset.x_unlabeled, dataset.y_unlabeled),
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=4)
    test_dataloader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=256,
                                 num_workers=4)

    class EarlyStopping:
        def __init__(self, patience=3):
            self.patience = patience
            self.best_loss = torch.inf
            self.steps_without_improvement = 0
        def check_stop(self, loss_val):
            if loss_val >= self.best_loss:
                self.steps_without_improvement += 1
                if self.steps_without_improvement > self.patience:
                    return True
            else:
                self.steps_without_improvement = 0
                self.best_loss = loss_val
            return False

    early_stop = EarlyStopping()
    iterator = tqdm(range(max_epochs), disable=disable_progess_bar)
    for e in iterator:
        for batch_x, batch_y in train_dataloader:
            yHat = model(batch_x)
            loss_value = loss(yHat, batch_y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        # early stopping on test
        with torch.no_grad():
            loss_sum = 0.0
            total = 0.0
            correct = 0.0
            for batch_x, batch_y in test_dataloader:
                yHat = model(batch_x)
                predicted = torch.argmax(yHat, dim=1)
                total += batch_y.size(0)
                correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
                class_loss = loss(yHat, torch.argmax(batch_y.long(), dim=1))
                loss_sum += class_loss.detach().cpu().numpy()
            if early_stop.check_stop(loss_sum):
                print(f"Early stop after {e} epochs")
                break
            iterator.set_postfix({"val loss": loss_sum, "val acc": correct / total})
    accuracy = correct / total
    return accuracy
