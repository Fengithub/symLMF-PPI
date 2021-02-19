"""
Reference: 
Wang, Y.; You, Z.-H.; Yang, S.; Li, X.; Jiang, T.-H.; Zhou, X., A High Efficient Biological Language Model 
for Predicting Protein⁻Protein Interactions. Cells 2019, 8, 122.

code modified based on the originial code available at: https://figshare.com/s/b35a2d3bf442a6f15b6e
"""

from gensim.models import Word2Vec
from keras.models import Model
from keras.models import load_model
from keras.layers import Conv1D, MaxPooling1D, Input, Dense, Flatten, GRU, Activation, Dropout,concatenate
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_sentence_vector(sen_words: list, model: Word2Vec):
    '''
    1. Representing sentence vector.
    '''
    sen_vec = 0
    counter = 0
    for word in sen_words:
        try:
            w2v = model.wv.word_vec(word)
            counter = counter + 1
        except KeyError:
            w2v = 0
        sen_vec = sen_vec + w2v

    if counter == 0:
        sen_vec = np.zeros([2048, ])  
    else:
        sen_vec = sen_vec / counter

    return sen_vec

def make_data(pos_pa_file, pos_pb_file, neg_pa_file, neg_pb_file):
    '''
    2.Generating train data set
    x shape:[num,dim],y shape:[num,1]
    '''

    model = Word2Vec.load('./word2vec_model/w2v_2048_extended')
    temp_x = []

    with open(pos_pa_file, encoding='utf-8', mode='r') as ta:
        line_num = len(ta.readlines())    

    with open(pos_pa_file, encoding='utf-8', mode='r') as ta, \
            open(neg_pa_file, encoding='utf-8', mode='r') as ta1, \
            open(pos_pb_file, encoding='utf-8', mode='r') as tb, \
            open(neg_pb_file, encoding='utf-8', mode='r') as tb1:

        for _ in range(line_num):
            ta_p = ta.readline()
            ta_p_vector = get_sentence_vector(ta_p.strip().split(), model)
            tb_p = tb.readline()
            tb_p_vector = get_sentence_vector(tb_p.strip().split(), model)
            ab_vector = np.concatenate((ta_p_vector, tb_p_vector))
            temp_x.append(ab_vector)
        positive_num = len(temp_x)

        for k in range(line_num):
            ta1_p = ta1.readline()
            ta1_p_vector = get_sentence_vector(ta1_p.strip().split(), model)
            tb1_p = tb1.readline()
            tb1_p_vector = get_sentence_vector(tb1_p.strip().split(), model)
            a1b1_vector = np.concatenate((ta1_p_vector, tb1_p_vector))
            temp_x.append(a1b1_vector)
        negative_num = len(temp_x) - positive_num

        positive_y = np.ones([positive_num, 1])
        negative_y = np.zeros([negative_num, 1])
        y = np.concatenate((positive_y, negative_y))
        x = np.array(temp_x)

    return x, y

def cnn(max_feature, filters, kernel_size, pool_size, strides):
    digit_input = Input(shape=(1, max_feature))
    x = Conv1D(filters, kernel_size, padding="same")(digit_input)
    x = Conv1D(filters, kernel_size, padding="same")(x)
    x = Dropout(0.5)(x)
    x = PReLU()(x)
    x = MaxPooling1D(pool_size=pool_size, strides=strides, padding="same")(x)

    y = Conv1D(32, kernel_size, padding="same")(digit_input)
    y = Dropout(0.5)(y)
    y = PReLU()(y)
    y = MaxPooling1D(pool_size=pool_size, strides=strides, padding="same")(y)

    k = Conv1D(128, kernel_size, padding="same")(digit_input)
    k = Dropout(0.5)(k)
    k = PReLU()(k)
    k = MaxPooling1D(pool_size=pool_size, strides=strides, padding="same")(k)

    z = concatenate([x,y,k])
    z = Flatten()(z)

    out = Dense(1, activation='sigmoid')(z)
    model = Model(digit_input, out)
    model.summary()

    return model

def cnn_cv_scan(fold_idx, pos_pa_file, pos_pb_file, neg_pa_file, neg_pb_file, data_file, predict_file):
    # define parameters
    patience= 30
    model_save_path='./patience_'+ str(patience)+ '_human'

    '''data'''
    max_feature = 4096
    batch_size = 64 

    '''convolution layer'''
    filters = 64
    kernel_size = 2
    pool_size = 2
    strides = 1

    # num_classes = 2
    epochs = 50

    # create model
    model = cnn(max_feature, filters, kernel_size, pool_size, strides)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.save(model_save_path)

    # get data
    x, y = make_data(pos_pa_file, pos_pb_file, neg_pa_file, neg_pb_file)
    x = x.reshape(-1, 1, max_feature)

    # 5cv
    skf = StratifiedKFold(n_splits=5, random_state=20210127, shuffle=True)
    skf_list = list(skf.split(x,y))
    train_index, test_index = skf_list[fold_idx]
    x_train = x[train_index]
    x_test = x[test_index]

    # turn to np.array
    x_train = np.array(x_train)
    x_test  = np.array(x_test)

    y_train = y[train_index]
    y_test = y[test_index]

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, callbacks=[early_stopping])

    test_pred = model.predict(x_test, verbose = 1)
    test_pred = test_pred.reshape(1,-1)
    test_pred = test_pred.tolist()
    test_pred = test_pred[0]

    y_test = y_test.reshape(1,-1)
    y_test = y_test.tolist()
    y_test = y_test[0]

    data_df = pd.read_csv(data_file, sep = '\t')
    X_data = data_df[['Protein_A_idx', 'Protein_B_idx']].to_numpy()
    X_test_data = [X_data[test_index, 0], X_data[test_index, 1]]
    pros_A, pros_B = X_test_data[0], X_test_data[1]
    df_predict = pd.DataFrame(pros_A, columns = ['Protein_A'])
    df_predict['Protein_B'] = pros_B
    df_predict['Label'] = y_test
    df_predict['Predicted_Score'] = test_pred
    df_predict.to_csv(predict_file, header = True, index = False, sep = '\t')

def run_cnn(dataset, org):
    data_file = '../../datasets/' + dataset + '/PPI_' + org + '_2019.txt' 
    pos_pa_file = '../../datasets/' + dataset + '/' + org + '_positive_pa.txt'
    pos_pb_file = '../../datasets/' + dataset + '/' + org + '_positive_pb.txt'
    neg_pa_file = '../../datasets/' + dataset + '/' + org + '_negative_pa.txt'
    neg_pb_file = '../../datasets/' + dataset + '/' + org + '_negative_pb.txt'
    folder = '../../datasets/' + dataset + '/cnn_' + org + '_results/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for fold_idx in [0,1,2,3,4]:
        predict_file = folder + str(fold_idx) + '_prediction.txt'
        cnn_cv_scan(fold_idx, pos_pa_file, pos_pb_file, neg_pa_file, neg_pb_file, data_file, predict_file)

if __name__ == '__main__':
    run_cnn('H.sapiens-extended', 'human')   ## 'S.cerevisiae', yeast
