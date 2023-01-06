from typing import Union, Any, Dict, Optional
import os
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from core.agent import BaseAgent
from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_numpy, to_torch_as

class TimeDistributedNet(Module):
    def __init__(self, state_space, n_hidden, num_layers=0):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        additional_layers = [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU()]*num_layers
        self.net = nn.Sequential(nn.Linear(state_space, n_hidden),
                                 nn.LeakyReLU(),
                                 *additional_layers,
                                 nn.Linear(n_hidden, 1),)
    def forward(self, input, *args, **kwargs):
        input = torch.as_tensor(input, device=self.device, dtype=torch.float32)
        return self.net(input)

class TianTimeDistributedNet(TimeDistributedNet):

    def forward(self, input, *args, **kwargs):
        logits = super().forward(input, *args, **kwargs )
        logits = logits.squeeze(dim=-1) # get rid of time distributed dimensions
        return logits, None # no hidden state


class SurrogatePolicy(BasePolicy):

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        clip_loss_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        self._gamma = 0.0
        self._n_step = 1
        self._iter = 0
        self._clip_loss_grad = clip_loss_grad


    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        self.optim.zero_grad()
        batch.to_torch()
        q = self(batch).logits
        # q = q[np.arange(len(q)), batch.act]
        q = torch.squeeze(q)
        returns = to_torch_as(batch.rew.flatten(), q)
        td_error = returns - q

        weight = to_torch_as(batch.pop("weight", 1.0), q)
        if self._clip_loss_grad:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(y, t, reduction="mean")
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}


    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                **kwargs: Any) -> Batch:
        obs = batch.obs
        q = self.model(obs, state=state, info=batch.info)
        q = q[0] # we don't deal with hidden states
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=q, act=act, state=None)


    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps


    def exploration_noise(
        self,
        act: Union[np.ndarray, Batch],
        batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act


class SAR(BaseAgent):
    """
    Top-Level Shell for the framework
    """

    def __init__(self, file="sar_13_12_22.pth", device=None):
        super().__init__()
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.file = file
        chkpt_path = os.path.join("agents/checkpoints", file)
        if not os.path.exists(chkpt_path):
            raise ValueError(f"Checkpoint {chkpt_path} does not exist")
        state_shape = 16
        hidden_sizes = 3*72
        net = TianTimeDistributedNet(state_shape, hidden_sizes).to(device)
        _ = torch.optim.Adam(net.parameters())
        agent = SurrogatePolicy(net, _)
        agent.load_state_dict(torch.load(chkpt_path, map_location=device))
        agent.model.device = device
        agent.model = agent.model.to(device)
        agent.device = device
        self.agent = agent.to(device)


    @classmethod
    def create_state_callback(cls, state_ids: list[int],
                              x_unlabeled: Tensor,
                              x_labeled: Tensor, y_labeled: Tensor,
                              per_class_instances: dict,
                              budget:int, added_images:int,
                              initial_test_acc:float, current_test_acc:float,
                              classifier: Module, optimizer: Optimizer) -> Union[Tensor, dict]:
        with torch.no_grad():
            sample_x = x_unlabeled[state_ids]
            sample_features = SAR._get_sample_features(sample_x, classifier, y_labeled.shape[1])
            interal_features = SAR._get_internal_features(initial_test_acc, current_test_acc, added_images, budget)
            interal_features = interal_features.unsqueeze(0).repeat(len(sample_features), 1)
            state = torch.cat([sample_features, interal_features], dim=1)
            state = state.cpu()
        return state


    @classmethod
    def _get_internal_features(cls, initial_test_acc, current_test_accuracy, added_images, budget):
        current_acc = torch.Tensor([current_test_accuracy]).cpu()
        improvement = torch.Tensor([current_test_accuracy - initial_test_acc]).cpu()
        avrg_improvement = torch.divide(improvement, max(1, added_images))
        progress = torch.Tensor([added_images / float(budget)]).cpu()
        return torch.cat([current_acc, improvement, avrg_improvement, progress])


    @classmethod
    def _get_sample_features(cls, x, classifier, n_classes):
        eps = 1e-7
        # prediction metrics
        pred = classifier(x).detach()
        pred = torch.softmax(pred, dim=1)
        two_highest, _ = pred.topk(2, dim=1)

        entropy = -torch.mean(pred * torch.log(eps + pred) + (1+eps-pred) * torch.log(1+eps-pred), dim=1)
        bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
        hist_list = [torch.histc(p, bins=10, min=0, max=1) for p in pred]
        hist = torch.stack(hist_list, dim=0) / n_classes

        state = torch.cat([
            bVsSB.unsqueeze(1),
            entropy.unsqueeze(1),
            hist
        ], dim=1)
        return state.cpu()


    def predict(self, state:Union[Tensor, dict], greed:float=0.0) ->Tensor:
        data = Batch({
            "obs": torch.unsqueeze(state, dim=0),
            "truncated": False,
            "info": Batch(),
        })
        result = self.agent(data)
        return torch.from_numpy(result.act)

    def get_meta_data(self)->str:
        s = super().get_meta_data()
        return f"{s}\n" \
               f"Checkpoint: {self.file}"