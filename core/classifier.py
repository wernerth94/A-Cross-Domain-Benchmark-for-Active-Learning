from typing import Tuple
import math
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out, calculate_gain, _calculate_correct_fan
import torch.nn.functional as F
from torch import Tensor
from core.data import BaseDataset
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def kaiming_uniform_seeded(
    rng, tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
):

    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_seeded,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity)

    if 0 in tensor.shape:
        print("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=rng)



class SeededLinear(nn.Linear):
    def __init__(self, model_rng, *args, **kwargs):
        self.model_rng = model_rng
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        kaiming_uniform_seeded(self.model_rng, self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            with torch.no_grad():
                self.bias.uniform_(-bound, bound, generator=self.model_rng)


class LinearModel(nn.Module):
    def __init__(self, model_rng, input_size:int, num_classes:int, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.out = SeededLinear(model_rng, input_size, num_classes)

    def _encode(self, x:Tensor)->Tensor:
        return x

    def forward(self, x:Tensor)->Tensor:
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x)
        return x


class DenseModel(nn.Module):
    def __init__(self, model_rng, input_size:int, num_classes:int, hidden_sizes:tuple, dropout=None, add_head=True):
        assert len(hidden_sizes) > 0
        super().__init__()
        self.dropout = dropout

        self.inpt = SeededLinear(model_rng, input_size, hidden_sizes[0])
        self.hidden = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            self.hidden.append(SeededLinear(model_rng, hidden_sizes[max(0, i - 1)], hidden_sizes[i]))
        if add_head:
            self.out = SeededLinear(model_rng, hidden_sizes[-1], num_classes)

    def _encode(self, x:Tensor)->Tensor:
        """
        The split bewteen encoding and prediction is important for agents that use latent features from the
        classifier like Coreset
        """
        if len(x.size()) == 4:
            # Pretext SimCLR workaround
            x = x.squeeze()
        x = self.inpt(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, training=self.training)
        for h_layer in self.hidden:
            x = h_layer(x)
            x = F.relu(x)
            if self.dropout is not None:
                x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x:Tensor)->Tensor:
        x = self._encode(x)
        if hasattr(self, "out"):
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
        self.hidden = nn.ModuleList()
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

    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     for i in range(len(self.hidden)):
    #         self.hidden[i] = self.hidden[i].to(*args, **kwargs)
    #     return self


def construct_model(model_rng, x_shape, n_classes, model_config, add_head=True) -> Tuple[nn.Module, int]:
        '''
        Constructs the model by name and additional parameters
        Returns model and its output dim
        '''
        model_type = model_config["type"].lower()
        dropout = model_config["dropout"] if "dropout" in model_config else None
        if model_type == "linear":
            return LinearModel(model_rng, x_shape[-1], n_classes, dropout), \
                   n_classes
        elif model_type == "resnet18":
            from core.resnet import ResNet18
            if "dropout" in model_config:
                # TODO
                return ResNet18(num_classes=n_classes, in_channels=x_shape[0],
                                add_head=add_head), \
                       n_classes if add_head else 512
            else:
                return ResNet18(num_classes=n_classes, in_channels=x_shape[0],
                                add_head=add_head), \
                       n_classes if add_head else 512
        elif model_type == "mlp":
            return DenseModel(model_rng,
                              input_size=x_shape[-1],
                              num_classes=n_classes,
                              hidden_sizes=model_config["hidden"],
                              dropout=dropout,
                              add_head=add_head), \
                   n_classes if add_head else model_config["hidden"][-1]
        else:
            raise NotImplementedError



def fit_and_evaluate(dataset:BaseDataset,
                     model_rng,
                     disable_progess_bar:bool=False,
                     max_epochs:int=4000):

    from core.helper_functions import EarlyStopping
    loss = nn.CrossEntropyLoss()
    model = dataset.get_classifier(model_rng)
    model = model.to(dataset.device)
    optimizer = dataset.get_optimizer(model)

    train_dataloader = DataLoader(TensorDataset(dataset.x_train, dataset.y_train),
                                  batch_size=dataset.classifier_batch_size,
                                  shuffle=True, num_workers=4)
    val_dataloader = DataLoader(TensorDataset(dataset.x_val, dataset.y_val), batch_size=512)
    test_dataloader = DataLoader(TensorDataset(dataset.x_test, dataset.y_test), batch_size=512)
    all_accs = []
    early_stop = EarlyStopping(patience=40)
    iterator = tqdm(range(max_epochs), disable=disable_progess_bar)
    for e in iterator:
        model.train()
        for batch_x, batch_y in train_dataloader:
            yHat = model(batch_x)
            loss_value = loss(yHat, batch_y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        # early stopping on test
        with torch.no_grad():
            model.eval()
            loss_sum = 0.0
            for batch_x, batch_y in val_dataloader:
                yHat = model(batch_x)
                class_loss = loss(yHat, torch.argmax(batch_y.long(), dim=1))
                loss_sum += class_loss.detach().cpu().numpy()
            if early_stop.check_stop(loss_sum):
                print(f"Early stop after {e} epochs")
                break

        correct = 0.0
        test_loss = 0.0
        for batch_x, batch_y in test_dataloader:
            yHat = model(batch_x)
            predicted = torch.argmax(yHat, dim=1)
            correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
            class_loss = loss(yHat, torch.argmax(batch_y.long(), dim=1))
            test_loss += class_loss.detach().cpu().numpy()
        test_acc = correct / len(dataset.x_test)
        all_accs.append(test_acc)
        iterator.set_postfix({"test loss": "%1.4f"%test_loss, "test acc": "%1.4f"%test_acc})
    return all_accs
