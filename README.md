# Active Learning Benchmark

## Dependencies
- matplotlib
- Pandas
- PyTorch
- torchvision
- sklearn
- tianshou
- (gym)

## Requirements
- Datasets from different modalities / areas
  - vector data
  - image data
  - text
  - timeseries?
- datasets are selected based on potential for complex AL algorithms
  - upper bound is computed
  - regret of heuristic methods is computed (relative to the upper bound)
- fine-tuned classifiers for each dataset
- two usecases: same domain / domain transfer

## Baselines
- Uncertainty Sampling
  - Entropy
  - Margin
- Coreset
- BALD Dropout (MC Dropout for Uncertainty)
- ActiveLearningByLearning (ALBL)
- BadgeSampling

## Structure
### Dataset
Each dataset class needs to inherit from BaseDataset and implement a set of functions:
- `__init__(...)`: Sets hyperparameters for this dataset:
  - budget: AL Budget
  - initial_points_per_class: Size of the seed set per class
  - classifier_batch_size: Batch size for classifier training
  - data_file: name of the file that will hold the normalized and processed data
  - cache_folder: location for downloaded and processed files
- `_download_data()`: Automatically downloads the data source files into self.cache_folder and stores the data in self.x_train, self.y_train, self.x_test and self.y_test. <br>
  Helper functions for processing the data:
  - postprocess_torch_dataset
  - postprocess_svm_data
- `_normalize_data()`: normalizes the values stored in self.x_train and self.x_test
- `get_classifier(hidden_dims)`: constructs and returns the classifier for this dataset. hidden_dims sets the default size of the classifier and acts as a hyperparameter.
The returned object is a torch.Module
- `get_optimizer(model)`: constructs and returns the optimizer for the classifier.
- (optional) `get_meta_data()`: can be overwritten to save some meta information that concerns the dataset, like the source or version

### Agent
Each agent class needs to inherit from BaseAgent and implement a set of functions:
- `__init__()`: sets hyperparameters for the agent, like a checkpoint or number of clusters, etc.
- `predict(state, greed)`: implements the forward pass of the agent. Receives a state according to create_state_callback (see below) and a greed value.
The agent computes a scalar score value for each input (along the batch dimension).
The greed value is a hyperparameter for RL based agents and should not be set by the framework.
- `create_state_callback(cls, ...)` (classmethod): This function is used by the environment to construct the specific state for each agent.
This function has access to all available information in the environment excluding the test set and the (hidden) labels of the currently unlableled data.
It returns a single 2D tensor of size len(state_ids) x state_space.
- (optional) `get_meta_data()`: can be overwritten to save some meta information that concerns the agent, like the checkpoint or other hyperparameters

### Run Script
A run script can execute any number of agents and datasets. \
It implements the basic reinforcement learning flow:
```python
done = False
state = env.reset()
while not done:
    action = agent.predict(state)
    state, reward, done, truncated, info = env.step(action.item())
```
And wraps the environment into a logging context manager:
```python
with EnvironmentLogger(env, log_path, util.is_cluster) as env:
```
The cross-validation runs should be set to at least 50. \
An example can be found in `al_benchmark/eval_splice.py`

## Datasets

|        | One Domain   | Domain Transfer |
|--------|--------------|-----------------|
| Vector | splice, dna  |                 |
| Image  | cifar10/100  | office          |
| Text   |              |                 |
