# A Cross Domain-Benchmark for Active Learning
This repository holds the source code for the NeurIPS 24 submission with the same name. \
This work is available under the [CC BY License](https://creativecommons.org/licenses/by/4.0/)

## Dependencies
Python >= 3.10 

Via pip:
- torch
- torchvision
- gym
- matplotlib
- Pandas
- scikit-learn
- faiss-cpu
- nltk (additional download for `nltk.word_tokenize` in News/TopV2 needed)
- PyYAML
- batchbald-redux
- ray\[tune\] (Optional)

## Quick Start
[Optional] Pre-download all datasets `download_all_datasets.py --data_folder <your_folder>`\
`--data_folder` sets the folder, where dataset files will be downloaded to \
You can run an evaluation with `evaluate.py --data_folder "<my_folder>" --agent <name> --dataset <name> --query_size <int>`\
Available Agents:
- `random`
- `margin`
- `entropy`
- `coreset` (CoreSet Greedy)
- `typiclust`
- `bald`
- `badge`
- `coregcn`
- `dsa`
- `lsa`

Available Datasets:
- `splice`
- `dna`
- `usps`
- `cifar10`
- `fashion_mnist`
- `mnist`
- `topv2`
- `news`

## Results
All generated results tables can be found in `results/`\
`macro_` tables are aggregated by domain \
`micro_` tables are per dataset

## Visualizations
All graphics from the paper are generated via the two notebooks `eval_plots.ipynb` and `other_plots.ipynb`.

## Parallel Runs
Parallelism is controlled by two parameters: `run_id`(default 1) and `restarts`(default 50)\
This starts one run with seed 1 that sequentially executes the evaluation 50 times. \
For full parallelism set `restarts` to 1 and execute 50 runs with increasing `run_ids`\
This will automatically collect the results after each finished run and store it in `<dataset>/<query_size>/<agent>/accuracies.csv`

Here is an example how to run 6 seeded runs with three different levels of parallelism \
![](doc/img/parallel_runs_example.png)

## Structure
### Dataset
Each dataset class needs to inherit from BaseDataset and implement a set of functions:
- `__init__()`: Sets hyperparameters for this dataset:
  - data_file: name of the file that will hold the preprocessed data
  - cache_folder: location for downloaded and processed files
- `_download_data()`: Automatically downloads the data source files into self.cache_folder, stores the data in `self.x_train, self.y_train, self.x_test and self.y_test` and normalizes `self.x_train` and `self.x_test`. <br>
- `load_pretext_data()`: Loads the version of the data that can be used for the pretext task, like SimCLR
- `get_pretext_transforms()`: Returns PyTorch data transforms for pretext training
- `get_pretext_validation_transforms()`: Returns PyTorch data transforms for pretext training
- (optional) `inject_config()`: Can be used to force some properties in the config
- (optional) `get_meta_data()`: can be overwritten to save some meta information that concerns the dataset, like the source or version

### Agent
Each agent class needs to inherit from BaseAgent and implement a set of functions:
- `__init__()`: sets hyperparameters for the agent, like a model-checkpoint or number of clusters, etc.
- `predict(state, x_unlabeled, ...)`: implements the forward pass of the agent.
Receives the full state with all available information
The agent computes its score and return the index/indices of x_unlabeled that are selected for labeling
- (optional) `inject_config()`: Can be used to force some properties in the config, i.e. dropout for BALD
- (optional) `get_meta_data()`: can be overwritten to save some meta information that concerns the agent, like the checkpoint or other hyperparameters

### Run Scripts
The main run script is called `evaluate.py`. \
It implements the basic reinforcement learning flow and wraps the environment into a logging context manager:
```python
with core.EnvironmentLogger(env, log_path, util.is_cluster) as env:
    done = False
    dataset.reset()
    state = env.reset()
    iterations = math.ceil(env.env.budget / args.query_size)
    iterator = tqdm(range(iterations), miniters=2)
    for i in iterator:
        action = agent.predict(*state)
        state, reward, done, truncated, info = env.step(action)
        iterator.set_postfix({"accuracy": env.accuracies[1][-1]})
        if done or truncated:
            # triggered when sampling batch_size is >1
            break
```
The run script will collect all intermediate results and aggregate them into one `accuracies.csv` and `losses.csv` per experiment.

### Other Scripts
- `evaluate_oracle.py` executes the greedy oracle algorithm or a dataset
- `compute_upper_bound.py` uses the full dataset to compute the upper bound for a dataset
- `train_encoder.py` executes the pretext task for a dataset and saves a checkpoint for the encoder model
- `ray_tune.py` optimizes the hyperparameters for one of three tasks:
  1. Normal classification
  2. Embedded classification
  3. Pretext 


### Config Template
```yaml
dataset: # general settings for un-encoded data
  budget: 10000
  classifier_fitting_mode: finetuning # finetuning or from_scratch
  initial_points_per_class: 100 # seed set size
  classifier_batch_size: 64 # batch size for training the classifier
  validation_split: 0.04 # size of the validation set in percentage

classifier: # classifier architecture for un-encoded data
  type: Resnet18

optimizer: # optimizer settings for un-encoded data
  type: NAdam
  lr: 0.001
  weight_decay: 0.0

dataset_embedded: # general settings for encoded data
  encoder_checkpoint: encoder_checkpoints/cifar10_27.03/model_seed1.pth.tar
  budget: 450
  classifier_fitting_mode: from_scratch
  initial_points_per_class: 1
  classifier_batch_size: 64

classifier_embedded: # classifier architecture for encoded data
#  type: MLP
#  hidden: [24, 12]
  type: Linear

optimizer_embedded: # optimizer settings for encoded data
  # Linear
  type: NAdam
  lr: 0.00171578341563099
  weight_decay: 2.38432342659786E-05
  # MLP
#  type: Adam
#  lr: 0.00422210204014432
#  weight_decay: 1.62121435184421E-08

# Settings for the Pretext Task (SimCLR)
# This is used for creating the encoder checkpoint that encodes the encoded data
pretext_encoder: 
  type: Resnet18
  feature_dim: 128

pretext_optimizer:
  type: SGD
  lr: 0.4
  nesterov: False
  weight_decay: 0.0001
  momentum: 0.9
  lr_scheduler: cosine
  lr_scheduler_decay: 0.1

pretext_clr_loss:
  temperature: 0.1

pretext_training:
  batch_size: 512
  epochs: 500
```
