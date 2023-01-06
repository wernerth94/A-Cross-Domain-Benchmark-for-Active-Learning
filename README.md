# Active Learning Benchmark

## Requirements
- two usecases: same domain / domain transfer
- Datasets from different modalities / areas
  - vector data
  - image data
  - text
  - timeseries?
- datasets are selected based on potential for complex AL algorithms
  - upper bound is computed
  - regret of heuristic methods is computed (relative to the upper bound)
- fine-tuned classifiers for each dataset
- callback function for custom state spaces
- different measurements of improvement
  - F1-Score
  - Accuracy
- different reward functions
  - improvement
  - absolute

## Baselines
- Coreset

## Datasets
### Image
- cifar10
- office (https://openreview.net/pdf?id=p98WJxUC3Ca)

### Vector
- splice
- dna

### Text

### Timeseries

### Graphs?

## Data Matrix

|        | One Domain  | Domain Transfer |
|--------|-------------|-----------------|
| Vector | splice, dna |                 |
| Image  | cifar10     | office          |
| Text   |             |                 |
