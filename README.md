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

nmtf.py: code for nonnegative matrix tri-factorization model  (Reference: Wang, H.; Huang, H.; Ding, C.; Nie, F., Predicting protein-protein interactions from multimodal biological data sources via nonnegative matrix tri-factorization. J Comput Biol 2013, 20, 344-58.)  

helper_functions.py: functions helps to process data, train model and evaluate performance  

cv_mf.py: demo of performing cross-validation  

hide_run.py: demo of train a model with hidding part of the input data  

train_model.py: demo of how to train the final model  

=============================================  

Top_predictions

Top 1000 prediction results for symLMF and SPRINT, as described in the manuscript  


=============================================  
cnn/  
Code used to perform CNN-Bio2vec method for performance comparison  

Reference: 
Wang, Y.; You, Z.-H.; Yang, S.; Li, X.; Jiang, T.-H.; Zhou, X., A High Efficient Biological Language Model 
for Predicting Protein⁻Protein Interactions. Cells 2019, 8, 122.

code modified based on the originial code available at: https://figshare.com/s/b35a2d3bf442a6f15b6e


=============================================  
dbn/  
Code used to perform DNN-Res2vec method for performance comparison 

Reference: Yao, Y.; Du, X.; Diao, Y.; Zhu, H., An integration of deep learning with feature embedding
for protein-protein interaction prediction. PeerJ 2019, 7, e7126.

code modified based on the originial code available on Github: https://github.com/xal2019/DeepFE-PPI  
