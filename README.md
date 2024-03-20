# Meta-IGCN: Interpretable Graph Convolutional Network for Alzheimer's Disease Diagnosis Based on a Meta-learning Paradigm
## Introduction
Meta-IGCN is an interpretable graph convolutional model based on the meta-learning paradigm for early diagnosis of Alzheimer's disease (AD). We utilize weighting and dimensionality reduction to process the functional connectivity data obtained from rs-fMRI, enhancing interpretability and improving storage and training efficiency. By leveraging the meta-learning paradigm, we sample subjects to form numerous tasks, thereby maximizing data utilization. In addition, the meta-learning paradigm enables the model to adapt to new tasks quickly and realizes independent testing of graph convolutional networks. Source code for Meta-IGCN: an interpretable method for early diagnosis of Alzheimer's disease by meta-learning paradigm.
## Files
### utils.py
This file undergoes the most basic data processing, such as obtaining data, dividing data sets, and so on.
### se_ae.py
The SE-AE block appears in this section, which performs the weighting and dimensionality reduction.
### meta.py
Meta-learning paradigm.
### models.py
Models required for the entire Meta-IGCN process.
### train.py
We train and test model using 5-fold cross-validation.
## Usage
Using Meta-IGCN, you can first obtain data features, labels, etc., through *se_ae.py*. Then the subjects' features after processing are accessible through the SE-AE block. Finally, training and testing the model are carried out in *train.py*. It is worth noting that random numbers may vary on different devices.
## Acknowledgment
Many thanks to [pygcn](https://github.com/tkipf/pygcn), [MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch), [Meta-GNN](https://github.com/ChengtaiCao/Meta-GNN), [AutoMetricGNN](https://github.com/SJTUBME-QianLab/AutoMetricGNN), [SENet](https://github.com/hujie-frank/SENet), [zhihu](https://zhuanlan.zhihu.com/p/625085766) and their authors. This code is developed on the code base of the above works.
