# Active Learning Benchmark

## Dependencies
- tianshou
- torchvision
- matplotlib
- Pandas
- sklearn

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
- `_download_data()`: Automatically downloads the data source files into self.cache_folder, stores the data in self.x_train, self.y_train, self.x_test and self.y_test and normalizes self.x_train and self.x_test. <br>
  Helper functions for processing the data:
  - `postprocess_torch_dataset`
  - `postprocess_svm_data`
  - `convert_to_channel_first`
- `get_classifier(hidden_dims)`: constructs and returns the classifier for this dataset. hidden_dims sets the default size of the classifier and acts as a hyperparameter.
The returned object is a torch.Module
- `get_optimizer(model)`: constructs and returns the optimizer for the classifier.
- (optional) `get_meta_data()`: can be overwritten to save some meta information that concerns the dataset, like the source or version

### Agent
Each agent class needs to inherit from BaseAgent and implement a set of functions:
- `__init__()`: sets hyperparameters for the agent, like a checkpoint or number of clusters, etc.
- `predict(state, state_ids, ...)`: implements the forward pass of the agent. 
Receives the full state with all available information.
The agent computes a scalar score value for each data id in state_ids and picks the instance as it sees fit.
- (optional) `get_meta_data()`: can be overwritten to save some meta information that concerns the agent, like the checkpoint or other hyperparameters

### Run Script
A run script can execute any number of agents and datasets. \
It implements the basic reinforcement learning flow:
```python
done = False
state = env.reset()
dataset.reset()
while not done:
    action = agent.predict(*state)
    state, reward, done, truncated, info = env.step(action.item())
```
And wraps the environment into a logging context manager:
```python
with EnvironmentLogger(env, log_path, util.is_cluster) as env:
```
The EnvironmentLogger also handles restarts of the evaluation process, so the loop can be restarted any amount
of times for cross-validation. \
Currently I have three run scripts:
- `al_benchmark/evaluation.py`: evaluates an AL agent on a dataset
- `al_benchmark/evaluate_oracle.py`: evaluates the oracle on a dataset
- `al_benchmark/compute_upper_bound.py`: computes maximum performance on a dataset with all data available

## Datasets

|        | One Domain   | Domain Transfer |
|--------|--------------|-----------------|
| Vector | splice, dna  |                 |
| Image  | cifar10/100  | office          |
| Text   |              |                 |
