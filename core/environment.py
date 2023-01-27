from typing import Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gym
from core.data import BaseDataset

class ALGame(gym.Env):

    def __init__(self, dataset:BaseDataset,
                 labeled_sample_size,
                 create_state_callback:Callable,
                 case_weight_multiplier=0.0,
                 device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.dataset = dataset
        self.budget = dataset.budget
        self.sample_size = labeled_sample_size
        self.fitting_mode = dataset.class_fitting_mode
        self.loss = nn.CrossEntropyLoss()
        self.create_state_callback = create_state_callback
        self.case_weight_multiplier = case_weight_multiplier

        # set gym observation space and action space
        self.current_test_accuracy = 0.0
        state = self.reset()
        if isinstance(state, dict):
            self.observation_space = dict()
            for key, value in state.items():
                self.observation_space[key] = gym.spaces.Box(-np.inf, np.inf, shape=[value.size(1),])
        else:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=[state.size(1),])
        self.action_space = gym.spaces.Discrete(self.sample_size)
        self.spec = gym.envs.registration.EnvSpec("RlAl-v0", reward_threshold=np.inf, entry_point="ALGame")


    def reset(self, *args, **kwargs)->Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            self.n_interactions = 0
            self.added_images = 0
            self.classifier = self.dataset.get_classifier()
            self.classifier.to(self.device)
            self.initial_weights = self.classifier.state_dict()
            self.optimizer = self.dataset.get_optimizer(self.classifier)
            self.reset_al_pool()
            self.case_weights = np.array([1.0]*len(self.x_labeled))
            self.case_weights /= self.case_weights.sum()
        # first training of the model should be done from scratch
        self._fit_classifier(from_scratch=True)
        self.initial_test_accuracy = self.current_test_accuracy
        self.state_ids = np.random.choice(len(self.x_unlabeled), self.sample_size)
        return self.create_state()


    def _send_state_to_device(self, state):
        if isinstance(state, dict):
            result = {}
            for k,v in state.items():
                result[k] = v.to(self.device)
        else:
            result = state.to(self.device)
        return result


    def create_state(self):
        state = self.create_state_callback(self.state_ids,
                                           self.x_unlabeled,
                                           self.x_labeled, self.y_labeled,
                                           self.per_class_instances,
                                           self.budget, self.added_images,
                                           self.initial_test_accuracy, self.current_test_accuracy,
                                           self.classifier, self.optimizer)
        return self._send_state_to_device(state)


    def step(self, action:int):
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
            # add next case weight and re-normalize
            self.case_weights = np.concatenate([self.case_weights, np.array([sum(self.case_weights) * self.case_weight_multiplier])])
            self.case_weights /= self.case_weights.sum()

        # fit classification model
        reward = self.fit_classifier()
        # pick new sample for the next state
        self.state_ids = np.random.choice(len(self.x_unlabeled), self.sample_size)
        next_state = self.create_state()
        done = self.added_images >= self.budget
        truncated = False
        return next_state, reward, done, truncated, {}


    def _fit_classifier(self, epochs=50, from_scratch=False):
        if from_scratch:
            self.classifier.load_state_dict(self.initial_weights)

        if self.case_weight_multiplier > 0.0:
            sampler = torch.utils.data.WeightedRandomSampler(self.case_weights, self.dataset.classifier_batch_size)
            train_dataloader = DataLoader(TensorDataset(self.x_labeled, self.y_labeled),
                                          batch_size=self.dataset.classifier_batch_size,
                                          sampler=sampler, num_workers=4)
        else:
            train_dataloader = DataLoader(TensorDataset(self.x_labeled, self.y_labeled),
                                          batch_size=self.dataset.classifier_batch_size,
                                          shuffle=True, num_workers=4)
        test_dataloader = DataLoader(TensorDataset(self.dataset.x_test, self.dataset.y_test), batch_size=100,
                                     num_workers=4)

        lastLoss = torch.inf
        for e in range(epochs):
            for batch_x, batch_y in train_dataloader:
                yHat = self.classifier(batch_x)
                loss_value = self.loss(yHat, batch_y)
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
            # early stopping on test
            with torch.no_grad():
                loss_sum = 0.0
                total = 0.0
                correct = 0.0
                for batch_x, batch_y in test_dataloader:
                    yHat = self.classifier(batch_x)
                    predicted = torch.argmax(yHat, dim=1)
                    total += batch_y.size(0)
                    correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
                    class_loss = self.loss(yHat, torch.argmax(batch_y.long(), dim=1))
                    loss_sum += class_loss.detach().cpu().numpy()
                # early stop on test with patience of 0
                if loss_sum >= lastLoss:
                    break
                lastLoss = loss_sum
        accuracy = correct / total
        self.current_test_loss = loss_sum

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


    def reset_al_pool(self):
        self.x_labeled = self.dataset.x_labeled
        self.y_labeled = self.dataset.y_labeled
        self.x_unlabeled = self.dataset.x_unlabeled
        self.y_unlabeled = self.dataset.y_unlabeled
        self.per_class_instances = [self.dataset.initial_points_per_class] * self.dataset.n_classes


    def render(self, mode="human"):
        '''
        dummy implementation of Gym.render() for the Gym-Interface
        :param mode:
        :return:
        '''
        pass

    def get_meta_data(self)->str:
        return f"{str(self)} \n" \
               f"Sample Size: {self.sample_size}"