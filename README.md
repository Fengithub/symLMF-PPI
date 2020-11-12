# symLMF-PPI
Predicting large-scale protein-protein interactions using symmetric logistic matrix factorization   
Fen Pei1,2, Qingya Shi1, 3, Haotian Zhang1, Ivet Bahar1
1. Department of Computational and Systems Biology, School of Medicine, University of Pittsburgh, PA, 15213, USA
2. Drug Discovery Institute University of Pittsburgh, PA, 15213, USA
3. School of Medicine, Tsinghua University, Beijing, 100084, China  

This work is submitted to the Journal of Chemical Information and Modeling
============================================

This project uses the following dependencies:

Python 3.7.7  
Numpy 1.19.1  
Scikit-learn 0.23.1  
Tensorflow 2.2.0  
keras 2.4.3  

============================================

Dataset  

The datasets directory contains the directories of S.cerevisiae-benchmark, H. sapiens-benchmark, S.cerevisiae-extended, H.sapiens-extended dataset, brain, liver, neurodegenerative_disease, disease_of_metabolism in each directory, there are:  

The S. cerevisiae-benchmark dataset (Guo, et al., 2008): 2526 proteins, 5594 positive protein pairs and 5594 negative protein pairs.
The H. sapiens-benchmark dataset (Huang, et al., 2015): 2835 proteins, 3899 positive protein pairs and 4262 negative protein pairs.  
The S. cerevisiae-extended dataset: 5142 proteins, 56316 positive protein pairs and 56316 negative protein pairs.  
The H. sapiens-extended dataset: 14455 proteins, 285618 positive protein pairs and 285618 negative protein pairs.  
brain: 11167 proteins, 225200 positive protein pairs and 225200 negative protein pairs.  
liver: 10627 proteins, 218239 positive protein pairs and 218239 negative protein pairs.  
neurodegenerative_disease: 820 proteins, 5881 positive protein pairs and 5881 negative protein pairs.  
disease_of_metabolism: 1063 proteins, 5131 positive protein pairs and 5131 negative protein pairs.   

=============================================  

Code

mf_models.py: code for symmetric logistic matrix factorization (symLMF),  symmetric probabilistic matrix factorization (symPMF) and symmetric nonnegative matrix factorization (symNMF) models.  

nmtf.py: code for nonnegative matrix tri-factorization model  

helper_functions.py: functions helps to process data, train model and evaluate performance  

cv_mf.py: demo of performing cross-validation  

hide_run.py: demo of train a model with hidding part of the input data  

train_model.py: demo of how to train the final model  

