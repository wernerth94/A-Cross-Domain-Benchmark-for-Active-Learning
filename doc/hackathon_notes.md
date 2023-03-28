```commandline
conda deactivate
conda create --name al_bench
conda activate al_bench
pip freeze | xargs pip uninstall -y #purges old links of installed packages
pip install -r first.txt
pip install -r second.txt
conda install requests
```

```commandline
conda deactivate
conda create --name al_bench
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install tianshou gym pandas matplotlib sklearn -c conda-forge
```

# Implementing Baselines
- MC Dropout for Uncertainty ([Link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf))
- Learning Loss for Active Learning ([Link](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf))
- Learning to Sample ([Link](https://arxiv.org/pdf/1909.03585.pdf))

# Fixing the Oracle 
- It should also outperform the other approaches in late stages
  <img height="400" src="img/oracle_performance.png" width="500"/>

# Setting the Budget
When should we stop the experiment? \
When
- oracle/random archives 95%/98% of the oracle performance ?
- until "convergence"?
- 

# Planting Non-Reducable Noise in the Data
A good AL algorithm should be able to distinguish data-noise from model-noise that stems from underfitting and needs to focus on the model-noise and ignore the data-noise.


# Postprocessing
- deriving SeededConvLayer
- ToyData Surrogate Agent
- GPU Env for Surrogate Agent
- Skim other NIPS benchmark papers
- AL Eval: Towards Robust Reprocudible Active Learning Using Neural Networks
- TypiClust Code (unsupervised embeddings, retraining the embeddings)
- TypiClust Oracle
- HP config
- Rework the Pseudocode
  - <img height="300" src="img/tmp.jpeg" width="500"/>
- Active Few Shot Learning (Split classes into train classes and testing classes)
- 
### Done
- Checking seeding for all tabular datasets with all agents
