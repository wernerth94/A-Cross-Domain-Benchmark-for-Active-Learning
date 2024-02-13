from typing import Union
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gym
from core.data import BaseDataset
from core.helper_functions import EarlyStopping


class ALGame(gym.Env):

    def __init__(self, dataset: BaseDataset,
                 pool_rng: np.random.Generator,
                 model_seed: int,
                 data_loader_seed: int = 2023,
                 device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.pool_rng = pool_rng
        self.data_loader_seed = data_loader_seed
        self.model_rng = torch.Generator()
        self.model_rng.manual_seed(model_seed)
        self.data_loader_rng = torch.Generator()
        self.data_loader_rng.manual_seed(self.data_loader_seed)

        self.dataset = dataset
        self.budget = dataset.budget
        self.fitting_mode = dataset.class_fitting_mode
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.current_test_accuracy = 0.0

        # Create Mock values to satisfy Gym interface
        state = self.reset()
        if isinstance(state, dict):
            self.observation_space = dict()
            for key, value in state.items():
                self.observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=[len(value), ])
        else:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=[len(state), ])
        self.action_space = gym.spaces.Discrete(len(dataset.x_unlabeled))
        self.spec = gym.envs.registration.EnvSpec("RlAl-v0", reward_threshold=np.inf, entry_point="ALGame")


    def reset(self, *args, **kwargs) -> list:
        with torch.no_grad():
            self.n_interactions = 0
            self.added_images = 0
            self.classifier = self.dataset.get_classifier(self.model_rng)
            self.classifier = self.classifier.to(self.device)
            self.initial_weights = self.classifier.state_dict()
            self.optimizer = self.dataset.get_optimizer(self.classifier)
            self.reset_al_pool()
        # first training of the model should be done from scratch
        self._fit_classifier(from_scratch=True)
        self.initial_test_accuracy = self.current_test_accuracy
        return self.create_state()


    def create_state(self):
        """
        Collects all necessary information that is exposed to the acquisition functions
        """
        state = [self.x_unlabeled,
                 self.x_labeled, self.y_labeled,
                 self.per_class_instances,
                 self.budget, self.added_images,
                 self.initial_test_accuracy, self.current_test_accuracy,
                 self.classifier, self.optimizer]
        return state


    def step(self, action: list[int]):
        """
        Adds one or more samples to the labeled set and retrains the classifier
        """
        reward = 0

        self.n_interactions += len(action)
        self.added_images += len(action)
        action = sorted(action)[::-1] # add datapoints from last index to first
        for a in action:
            with torch.no_grad():
                self._add_point_to_labeled_pool(a)

            if self.fitting_mode == "finetuning":
                reward = self.fit_classifier()
        if self.fitting_mode == "from_scratch":
                reward = self.fit_classifier()
        next_state = self.create_state()
        done = self.added_images >= self.budget
        truncated = False
        return next_state, reward, done, truncated, {}


    def _fit_classifier(self, epochs=50, from_scratch=False):
        if from_scratch:
            self.classifier.load_state_dict(self.initial_weights)
            early_stop = EarlyStopping(patience=50)
        else:
            early_stop = EarlyStopping(patience=0)

        # If drop_last is True, the first iterations i < batch_size have no training data
        drop_last = self.dataset.classifier_batch_size < len(self.x_labeled)
        train_dataloader = DataLoader(TensorDataset(self.x_labeled, self.y_labeled),
                                      batch_size=self.dataset.classifier_batch_size,
                                      drop_last=drop_last,
                                      generator=self.data_loader_rng,
                                      shuffle=True)
        val_dataloader = DataLoader(TensorDataset(self.dataset.x_val, self.dataset.y_val), batch_size=512)
        test_dataloader = DataLoader(TensorDataset(self.dataset.x_test, self.dataset.y_test), batch_size=512)
        val_loss_list = [] # used for debugging only
        for e in range(epochs):
            self.classifier.train()
            for i, (batch_x, batch_y) in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                yHat = self.classifier(batch_x)
                loss_value = self.loss(yHat, batch_y)
                loss_value.backward()
                self.optimizer.step()

            # early stopping on validation
            with torch.no_grad():
                self.classifier.eval()
                loss_sum = 0.0
                for batch_x, batch_y in val_dataloader:
                    yHat = self.classifier(batch_x)
                    class_loss = self.loss(yHat, torch.argmax(batch_y.long(), dim=1))
                    loss_sum += class_loss.detach().cpu().numpy()

                val_loss_list.append(loss_sum)
                if early_stop.check_stop(loss_sum):
                    break
        self.current_test_loss = loss_sum

        # create reward on the testing set
        with torch.no_grad():
            self.classifier.eval()
            correct = 0.0
            for batch_x, batch_y in test_dataloader:
                yHat = self.classifier(batch_x)
                predicted = torch.argmax(yHat, dim=1)
                # count correct predictions
                correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
            accuracy = correct / len(self.dataset.x_test)
            reward = accuracy - self.current_test_accuracy
            self.current_test_accuracy = accuracy
        return reward


    def fit_classifier(self, epochs=100):
        if self.fitting_mode == "from_scratch":
            return self._fit_classifier(epochs, from_scratch=True)
        elif self.fitting_mode == "finetuning":
            return self._fit_classifier(epochs, from_scratch=False)
        elif self.fitting_mode == "one_epoch":
            return self._fit_classifier(1, from_scratch=False)
        else:
            raise ValueError(f"Fitting mode not recognized: {self.fitting_mode}")


    def _add_point_to_labeled_pool(self, datapoint_id):
        # keep track of the added images
        self.per_class_instances[int(torch.argmax(self.y_unlabeled[datapoint_id]).cpu())] += 1
        # add the point to the labeled set
        self.x_labeled = torch.cat([self.x_labeled, self.x_unlabeled[datapoint_id:datapoint_id + 1]], dim=0)
        self.y_labeled = torch.cat([self.y_labeled, self.y_unlabeled[datapoint_id:datapoint_id + 1]], dim=0)
        # remove the point from the unlabeled set
        self.x_unlabeled = torch.cat([self.x_unlabeled[:datapoint_id], self.x_unlabeled[datapoint_id + 1:]], dim=0)
        self.y_unlabeled = torch.cat([self.y_unlabeled[:datapoint_id], self.y_unlabeled[datapoint_id + 1:]], dim=0)


    def reset_al_pool(self):
        self.x_labeled = self.dataset.x_labeled
        self.y_labeled = self.dataset.y_labeled
        self.x_unlabeled = self.dataset.x_unlabeled
        self.y_unlabeled = self.dataset.y_unlabeled
        self.per_class_instances = [self.dataset.initial_points_per_class] * self.dataset.n_classes


    def render(self, mode="human"):
        """
        dummy implementation of Gym.render() for the Gym-Interface
        :param mode:
        :return:
        """
        pass


    def get_meta_data(self) -> str:
        return f"{str(self)} \n"


class OracleALGame(ALGame):
    """
    Custom environment for the oracle algorithm with modified step function.
    """

    def __init__(self, dataset: BaseDataset,
                 labeled_sample_size,
                 pool_rng: np.random.Generator,
                 model_seed: int,
                 data_loader_seed: int = 2023,
                 points_added_per_round=1,
                 device=None):
        super().__init__(dataset, pool_rng, model_seed, data_loader_seed, device)
        self.points_added_per_round = points_added_per_round
        self.starting_state_rng = np.random.default_rng(self.data_loader_seed)
        self.oracle_counter = 0
        self.sample_size = labeled_sample_size

    def _get_internal_state(self):
        initial_weights = copy.deepcopy(self.classifier.state_dict())
        initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        initial_test_loss = self.current_test_loss
        initial_test_acc = self.current_test_accuracy
        return (initial_weights, initial_optimizer_state,
                initial_test_loss, initial_test_acc)

    def _set_internal_state(self, state_tuple):
        self.classifier.load_state_dict(state_tuple[0])
        self.optimizer.load_state_dict(state_tuple[1])
        self.current_test_loss = state_tuple[2]
        self.current_test_accuracy = state_tuple[3]
        self.data_loader_rng.manual_seed(self.data_loader_seed_i)


    def _get_scores(self, state_ids)->torch.Tensor:
        scores = torch.zeros(len(state_ids))
        for act, i in enumerate(state_ids):
            with torch.no_grad():
                # restore initial states
                self._set_internal_state(self.initial_state)
                # add testing point to labeled pool
                self.x_labeled = torch.cat([self.x_labeled, self.x_unlabeled[i:i + 1]], dim=0)
                self.y_labeled = torch.cat([self.y_labeled, self.y_unlabeled[i:i + 1]], dim=0)
            reward = self.fit_classifier()
            with torch.no_grad():
                scores[act] = reward
                # remove the testing point
                self.x_labeled = self.x_labeled[:-1]
                self.y_labeled = self.y_labeled[:-1]
        return scores

    def _get_margin_scores(self, state_ids):
        with torch.no_grad():
            x_sample = self.x_unlabeled[state_ids]
            pred = self.classifier(x_sample).detach()
            pred = torch.softmax(pred, dim=1)
            two_highest, _ = pred.topk(2, dim=1)
            bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
            torch.unsqueeze(bVsSB, dim=-1)
            # action = torch.argmax(bVsSB, dim=0)
            # best_i = state_ids[action]
        return bVsSB

    def step(self, *args, **kwargs):
        """
        Custom step function that ignores any passed action and searches for the best point in a greedy fashion
        """
        # preserve the initial state for this iteration
        self.initial_state = self._get_internal_state()
        self.data_loader_seed_i = int(self.starting_state_rng.integers(1, 1000, 1)[0])
        replacement_needed = len(self.x_unlabeled) < self.sample_size
        state_ids = self.pool_rng.choice(len(self.x_unlabeled), self.sample_size, replace=replacement_needed)

        scores = self._get_scores(state_ids)
        chosen = torch.topk(scores, self.points_added_per_round).indices.tolist()
        # restore initial states one last time
        self._set_internal_state(self.initial_state)

        margin_scores = self._get_margin_scores(state_ids)
        margin_sorted_ids = torch.sort(margin_scores, descending=True).indices


        used_ids = []
        used_ids.extend(chosen) # deep copy
        for id in chosen:
            if scores[id] <= 0.0:
                # No point with positive impact was found. Defaulting to Margin sampling
                for m_id in margin_sorted_ids:
                    m_id = m_id.item()
                    if m_id not in used_ids:
                        id = m_id
                        used_ids.append(m_id)
                        break
            with torch.no_grad():
                self.per_class_instances[int(torch.argmax(self.y_unlabeled[id]).cpu())] += 1
                # add the point to the labeled set
                self.x_labeled = torch.cat([self.x_labeled, self.x_unlabeled[id:id + 1]], dim=0)
                self.y_labeled = torch.cat([self.y_labeled, self.y_unlabeled[id:id + 1]], dim=0)
                # remove the point from the unlabeled set
                self.x_unlabeled = torch.cat([self.x_unlabeled[:id], self.x_unlabeled[id + 1:]], dim=0)
                self.y_unlabeled = torch.cat([self.y_unlabeled[:id], self.y_unlabeled[id + 1:]], dim=0)
        reward = self.fit_classifier()
        self.added_images += 1
        done = self.added_images >= self.budget
        truncated, info = False, {"action": chosen}
        return self.create_state(), reward, done, truncated, info
