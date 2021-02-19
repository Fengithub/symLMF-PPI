"""
Reference:Yao, Y.; Du, X.; Diao, Y.; Zhu, H., An integration of deep learning with feature embedding 
for protein-protein interaction prediction. PeerJ 2019, 7, e7126.

code modified based on the originial code available on Github: https://github.com/xal2019/DeepFE-PPI
"""

import copy
import h5py
import os
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout
from tensorflow.keras.layers import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers import Input
from keras.optimizers import SGD
from utils import *

def merged_DBN(sequence_len):
    #left_model
    input_left = Input(shape = (18000,)) # for human is 18000, but for yeast is 17000
    model_left = Dense(2048, input_dim=sequence_len,activation='relu',kernel_regularizer=l2(0.01))(input_left)
    model_left = BatchNormalization()(model_left)
    model_left = Dropout(0.5)(model_left)

    model_left = Dense(1024, activation='relu',kernel_regularizer=l2(0.01))(model_left)
    model_left = BatchNormalization()(model_left)
    model_left = Dropout(0.5)(model_left)

    model_left = Dense(512, activation='relu',kernel_regularizer=l2(0.01))(model_left)
    model_left = BatchNormalization()(model_left)
    model_left = Dropout(0.5)(model_left)

    model_left = Dense(128, activation='relu',kernel_regularizer=l2(0.01))(model_left)
    model_left = BatchNormalization()(model_left)

    #right_model
    input_right = Input(shape = (18000,))  # for human is 18000, but for yeast is 17000
    model_right = Dense(2048, input_dim=sequence_len,activation='relu',kernel_regularizer=l2(0.01))(input_right)
    model_right = BatchNormalization()(model_right)
    model_right = Dropout(0.5)(model_right)

    model_right = Dense(1024, activation='relu',kernel_regularizer=l2(0.01))(model_right)
    model_right = BatchNormalization()(model_right)
    model_right = Dropout(0.5)(model_right)

    model_right = Dense(512, activation='relu',kernel_regularizer=l2(0.01))(model_right)
    model_right = BatchNormalization()(model_right)
    model_right = Dropout(0.5)(model_right)

    model_right = Dense(128, activation='relu',kernel_regularizer=l2(0.01))(model_right)
    model_right = BatchNormalization()(model_right)

    #together
    concatenated = concatenate([model_left, model_right])
    out = Dense(8, activation='relu',kernel_regularizer=l2(0.01))(concatenated)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Dense(2, activation='softmax')(out)

    model = Model([input_left, input_right], out)

    return model

def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])
        token_dataset.append(seq)

    return token_dataset

def pandding_J(protein, maxlen):
    padded_protein = copy.deepcopy(protein)
    for i in range(len(padded_protein)):
        if len(padded_protein[i]) < maxlen:
            for j in range(len(padded_protein[i]), maxlen):
                padded_protein[i] = padded_protein[i] + 'J'

    return padded_protein

## convert each protein to a vector
def residue_representation(wv, tokened_seq_protein, maxlen, size):
    represented_protein = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j] == 'J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))

    return np.array(represented_protein)

def protein_reprsentation(wv, pos_neg_protein_A, pos_neg_protein_B, maxlen, size):

    # padding
    padded_pos_neg_protein_A = pandding_J(pos_neg_protein_A, maxlen)
    padded_pos_neg_protein_B = pandding_J(pos_neg_protein_B, maxlen)

    # token
    token_padded_pos_neg_protein_A = token(padded_pos_neg_protein_A)
    token_padded_pos_neg_protein_B = token(padded_pos_neg_protein_B)

    # generate feature of pair A (vectors for protein A and B)
    feature_protein_A = residue_representation(wv, token_padded_pos_neg_protein_A, maxlen, size)
    feature_protein_B = residue_representation(wv, token_padded_pos_neg_protein_B, maxlen, size)

    feature_protein_AB = np.hstack((np.array(feature_protein_A), np.array(feature_protein_B)))

    return feature_protein_AB

def parse_data(data_file, fasta_file):
    data_df = pd.read_csv(data_file, sep = '\t')
    data_df['Interaction'] = data_df.apply(lambda x: 1 if x['Interaction'] > 0 else 0, axis = 1)
    fasta_df = pd.read_csv(fasta_file, sep = '\t')
    unip2fasta = fasta_df.set_index('Uniprot_ID')['Fasta'].to_dict()
    data_df['Fasta_A'] = data_df.apply(lambda x: unip2fasta[x['Uniprot_A']], axis = 1)
    data_df['Fasta_B'] = data_df.apply(lambda x: unip2fasta[x['Uniprot_B']], axis = 1)

    return data_df

def create_hdf5_file(protein_A_seq, protein_B_seq, maxlen, size, Y, filename):
    model_wv = Word2Vec.load('./word2vec_model/dbn_model.model')
    feature_protein_AB = protein_reprsentation(model_wv.wv, protein_A_seq, protein_B_seq, maxlen, size)

    # StandardScaler
    scaler = StandardScaler().fit(feature_protein_AB)
    feature_protein_AB = scaler.transform(feature_protein_AB)

    h5_file = h5py.File(filename, 'w')
    h5_file.create_dataset('dataset_x', data = feature_protein_AB)
    h5_file.create_dataset('dataset_y', data = Y)
    h5_file.close()

    return feature_protein_AB

def dbn_cv_scan(data_df, folder, batch_size, epochs, fold_idx, predict_file):
    """ 5-fold cross validation with parameters scanning """
    protein_A_seq = data_df['Fasta_A']
    protein_B_seq = data_df['Fasta_B']
    Y = data_df['Interaction'].to_numpy()
    y_labels = to_categorical(Y)

    maxlens = [900]  # for human is 900, for yeast is 850
    size = 20    # residue dimensionality

    for maxlen in maxlens:
        h5_filename = folder + str(maxlen) + '_' + str(size) + '.h5'
        if not os.path.isfile(h5_filename):
            feature_protein_AB = create_hdf5_file(protein_A_seq, protein_B_seq, maxlen, size, Y, h5_filename)
        else:
            f = h5py.File(h5_filename, 'r')
            feature_protein_AB = f['dataset_x']

        sequence_len = size * maxlen
        
        # 5cv
        skf = StratifiedKFold(n_splits=5, random_state=20210107, shuffle=True)
        skf_list = list(skf.split(feature_protein_AB, Y))
        train_index, test_index = skf_list[fold_idx]
        X_train_left = feature_protein_AB[train_index][:,0:sequence_len]
        X_train_right = feature_protein_AB[train_index][:,sequence_len:sequence_len*2]

        X_test_left = feature_protein_AB[test_index][:,0:sequence_len]
        X_test_right = feature_protein_AB[test_index][:,sequence_len:sequence_len*2]

        # turn to np.array
        X_train_left  = np.array(X_train_left)
        X_train_right  = np.array(X_train_right)

        X_test_left  = np.array(X_test_left)
        X_test_right  = np.array(X_test_right)

        X_train = [X_train_left,X_train_right]
        X_test= [X_test_left, X_test_right]

        y_train = y_labels[train_index]
        y_test = y_labels[test_index]
        
        model = merged_DBN(sequence_len)
        sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1)

        print('******   model created!  ******')

        test_pred = model.predict(X_test)
        auc = roc_auc_score(y_test[:,1], test_pred[:,1])
        aupr = average_precision_score(y_test[:,1], test_pred[:,1])

        label_predict_test = categorical_probas_to_classes(test_pred)
        tp_test,fp_test,tn_test,fn_test,accuracy_test, precision_test, sensitivity_test,recall_test, specificity_test, MCC_test, f1_score_test,_,_,_= calculate_performace(len(label_predict_test), label_predict_test, y_test[:,1])
        print(' ===========  test:'+ str(0))
        print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp_test,fp_test,tn_test,fn_test))
        print('\tacc=%0.4f,pre=%0.4f,rec=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f' % (accuracy_test, precision_test, recall_test, specificity_test, MCC_test, f1_score_test))
        print('\tauc=%0.4f,pr=%0.4f'%(auc,aupr))

        X_data = data_df[['Protein_A_idx', 'Protein_B_idx']].to_numpy()
        X_test_data = [X_data[test_index, 0], X_data[test_index, 1]]
        pros_A, pros_B = X_test_data[0], X_test_data[1]
        df_predict = pd.DataFrame(pros_A, columns = ['Protein_A'])
        df_predict['Protein_B'] = pros_B
        df_predict['Label'] = y_test[:,1]
        df_predict['Predicted_Score'] = test_pred[:,1]
        df_predict.to_csv(predict_file, header = True, index = False, sep = '\t')

def run_dbn(dataset, org):
    data_file = '../datasets/' + dataset + '/PPI_' + org + '_2019.txt' 
    fasta_file = '../datasets/' + dataset + '/unip2fasta_' + org + '.txt'
    folder = '../datasets/' + dataset + '/dbn_' + org + '_results/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    data_df = parse_data(data_file, fasta_file)
    batch_size = 256
    num_epoch = 35

    for fold_idx in [0,1,2,3,4]:
        predict_file = folder + str(fold_idx) + '_prediction.txt'
        dbn_cv_scan(data_df, folder, batch_size, num_epoch, fold_idx, predict_file)

if __name__ == '__main__':
    run_dbn('H.sapiens-extended', 'human')   ## 'S.cerevisiae', yeast
