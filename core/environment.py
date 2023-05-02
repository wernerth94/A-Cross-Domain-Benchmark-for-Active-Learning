import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gym
from core.data import BaseDataset


class ALGame(gym.Env):

    def __init__(self, dataset: BaseDataset,
                 labeled_sample_size: int,
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
        # torch.random.manual_seed(model_seed)

        self.dataset = dataset
        self.budget = dataset.budget
        self.sample_size = labeled_sample_size
        self.fitting_mode = dataset.class_fitting_mode
        self.loss = nn.CrossEntropyLoss().to(self.device)

        # set gym observation space and action space
        self.current_val_accuracy = 0.0
        state = self.reset()

        if isinstance(state, dict):
            self.observation_space = dict()
            for key, value in state.items():
                self.observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=[len(value), ])
        else:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=[len(state), ])
        self.action_space = gym.spaces.Discrete(self.sample_size)
        self.spec = gym.envs.registration.EnvSpec("RlAl-v0", reward_threshold=np.inf, entry_point="ALGame")

    def reset(self, *args, **kwargs) -> list:
        with torch.no_grad():
            self.n_interactions = 0
            self.added_images = 0
            self.classifier = self.dataset.get_classifier(self.model_rng)
            # classifier.custom_init(self.classifier, self.model_rng)
            self.classifier.to(self.device)
            self.initial_weights = self.classifier.state_dict()
            self.optimizer = self.dataset.get_optimizer(self.classifier)
            self.reset_al_pool()
        # first training of the model should be done from scratch
        self._fit_classifier(from_scratch=True)
        self.initial_test_accuracy = self.current_val_accuracy
        return self.create_state()

    def create_state(self):
        replacement_needed = len(self.x_unlabeled) < self.sample_size
        self.state_ids = self.pool_rng.choice(len(self.x_unlabeled), self.sample_size, replace=replacement_needed)
        state = [self.state_ids,
                 self.x_unlabeled,
                 self.x_labeled, self.y_labeled,
                 self.per_class_instances,
                 self.budget, self.added_images,
                 self.initial_test_accuracy, self.current_val_accuracy,
                 self.classifier, self.optimizer]
        return state

    def step(self, action: int):
        with torch.no_grad():
            self.n_interactions += 1
            self.added_images += 1
            datapoint_id = self.state_ids[action]
            # keep track of the added images
            self.per_class_instances[int(torch.argmax(self.y_unlabeled[datapoint_id]).cpu())] += 1
            # add the point to the labeled set
            self.x_labeled = torch.cat([self.x_labeled, self.x_unlabeled[datapoint_id:datapoint_id + 1]], dim=0)
            self.y_labeled = torch.cat([self.y_labeled, self.y_unlabeled[datapoint_id:datapoint_id + 1]], dim=0)
            # remove the point from the unlabeled set
            self.x_unlabeled = torch.cat([self.x_unlabeled[:datapoint_id], self.x_unlabeled[datapoint_id + 1:]], dim=0)
            self.y_unlabeled = torch.cat([self.y_unlabeled[:datapoint_id], self.y_unlabeled[datapoint_id + 1:]], dim=0)

        # fit classification model
        reward = self.fit_classifier()
        # pick new sample for the next state
        next_state = self.create_state()
        done = self.added_images >= self.budget
        truncated = False
        return next_state, reward, done, truncated, {}

    def _fit_classifier(self, epochs=50, from_scratch=False):
        if from_scratch:
            self.classifier.load_state_dict(self.initial_weights)

        drop_last = self.dataset.classifier_batch_size < len(self.x_labeled)
        train_dataloader = DataLoader(TensorDataset(self.x_labeled, self.y_labeled),
                                      batch_size=self.dataset.classifier_batch_size,
                                      drop_last=drop_last,
                                      generator=self.data_loader_rng,
                                      # num_workers=4, # dropped for CUDA compat
                                      shuffle=True)
        val_dataloader = DataLoader(TensorDataset(self.dataset.x_val, self.dataset.y_val), batch_size=512,
                                     # num_workers=4 # dropped for CUDA compat
                                     )
        test_dataloader = DataLoader(TensorDataset(self.dataset.x_test, self.dataset.y_test), batch_size=512,
                                     # num_workers=4 # dropped for CUDA compat
                                     )
        lastLoss = torch.inf
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
                # early stop on test with patience of 0
                if loss_sum >= lastLoss:
                    break
                lastLoss = loss_sum
        self.current_val_loss = loss_sum

        with torch.no_grad():
            self.classifier.eval()
            correct = 0.0
            for batch_x, batch_y in test_dataloader:
                yHat = self.classifier(batch_x)
                predicted = torch.argmax(yHat, dim=1)
                correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
            accuracy = correct / len(self.dataset.x_test)
            reward = accuracy - self.current_val_accuracy
            self.current_val_accuracy = accuracy
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
        return f"{str(self)} \n" \
               f"Sample Size: {self.sample_size}"


class OracleALGame(ALGame):
    def __init__(self, dataset: BaseDataset,
                 labeled_sample_size,
                 pool_rng: np.random.Generator,
                 model_seed: int,
                 data_loader_seed: int = 2023,
                 device=None):
        super().__init__(dataset, labeled_sample_size, pool_rng, model_seed, data_loader_seed, device)
        self.starting_state_rng = np.random.default_rng(self.data_loader_seed)
        self.oracle_counter = 0

    def _get_internal_state(self):
        initial_weights = copy.deepcopy(self.classifier.state_dict())
        initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        initial_val_loss = self.current_val_loss
        initial_val_acc = self.current_val_accuracy
        return (initial_weights, initial_optimizer_state,
                initial_val_loss, initial_val_acc)

    def _set_internal_state(self, state_tuple):
        self.classifier.load_state_dict(state_tuple[0])
        self.optimizer.load_state_dict(state_tuple[1])
        self.current_val_loss = state_tuple[2]
        self.current_val_accuracy = state_tuple[3]
        self.data_loader_rng.manual_seed(self.data_loader_seed_i)

    def step(self, *args, **kwargs):
        max_reward = 0.0
        best_i = -1
        best_action = -1
        # preserve the initial state for this iteration
        self.initial_state = self._get_internal_state()
        self.data_loader_seed_i = int(self.starting_state_rng.integers(1, 1000, 1)[0])
        for act, i in enumerate(self.state_ids):
            with torch.no_grad():
                # restore initial states
                self._set_internal_state(self.initial_state)
                # add testing point to labeled pool
                self.x_labeled = torch.cat([self.x_labeled, self.x_unlabeled[i:i + 1]], dim=0)
                self.y_labeled = torch.cat([self.y_labeled, self.y_unlabeled[i:i + 1]], dim=0)
            reward = self.fit_classifier()
            with torch.no_grad():
                if reward > max_reward:
                    max_reward = reward
                    best_i = i
                    best_action = act
                # remove the testing point
                self.x_labeled = self.x_labeled[:-1]
                self.y_labeled = self.y_labeled[:-1]
        # restore initial states one last time
        self._set_internal_state(self.initial_state)
        if max_reward == 0.0:
            # No point with positive impact was found. Defaulting to Margin sampling
            x_sample = self.x_unlabeled[self.state_ids]
            pred = self.classifier(x_sample).detach()
            pred = torch.softmax(pred, dim=1)
            two_highest, _ = pred.topk(2, dim=1)
            bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
            torch.unsqueeze(bVsSB, dim=-1)
            action = torch.argmax(bVsSB, dim=0)
            best_i = self.state_ids[action]
        with torch.no_grad():
            self.per_class_instances[int(torch.argmax(self.y_unlabeled[best_i]).cpu())] += 1
            # add the point to the labeled set
            self.x_labeled = torch.cat([self.x_labeled, self.x_unlabeled[best_i:best_i + 1]], dim=0)
            self.y_labeled = torch.cat([self.y_labeled, self.y_unlabeled[best_i:best_i + 1]], dim=0)
            # remove the point from the unlabeled set
            self.x_unlabeled = torch.cat([self.x_unlabeled[:best_i], self.x_unlabeled[best_i + 1:]], dim=0)
            self.y_unlabeled = torch.cat([self.y_unlabeled[:best_i], self.y_unlabeled[best_i + 1:]], dim=0)
        reward = self.fit_classifier()
        self.added_images += 1
        done = self.added_images >= self.budget
        truncated, info = False, {"action": best_action}
        return self.create_state(), reward, done, truncated, info
