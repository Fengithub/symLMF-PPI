Reference:Yao, Y.; Du, X.; Diao, Y.; Zhu, H., An integration of deep learning with feature embedding   
for protein-protein interaction prediction. PeerJ 2019, 7, e7126.  

code modified based on the originial code available on Github: https://github.com/xal2019/DeepFE-PPI  

1. Use get_sequences.py to get the sequences of human and yeast proteins, and classify the proteins into positive ones and negative ones.
2. Use get_words.py to convert each protein to a word.
3. Use cv_cnn.py to convert each protein to a vector, and then perform 5-fold cross-validation.  
