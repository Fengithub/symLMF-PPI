import numpy as np
import pandas as pd
import random
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import os
from sklearn.model_selection import StratifiedKFold
from mf_models import *
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def parse_data(data_file, binary, data_source):
    # load data
    if data_source == 'benchmark':
        data_df = pd.read_csv(data_file, sep = '\t', usecols = ['Protein_A_idx', 'Protein_B_idx', 'Interaction'], dtype = 'int32')
    else:
        data_df = pd.read_csv(data_file, sep = '\t', usecols = ['Protein_A_idx', 'Protein_B_idx', 'Interaction'])

    if data_source == 'tissue':
        data_df = data_df[['Protein_A_idx', 'Protein_B_idx', 'Interaction']] # re-order the columns

    if binary: # use binary input
        data_df['Interaction'] = data_df.apply(lambda x: 1 if x['Interaction'] > 0 else 0, axis = 1)

    # total number of proteins
    n_pros = len(set(data_df['Protein_A_idx'].tolist() + data_df['Protein_B_idx'].tolist()))

    X_data = data_df[['Protein_A_idx', 'Protein_B_idx']].to_numpy()
    y_val = data_df['Interaction'].to_numpy()

    return n_pros, X_data, y_val

def split_data(n_folds, X_data, y_val):
    skf = StratifiedKFold(n_splits=n_folds, random_state=20200524, shuffle=True)
    skf_list = list(skf.split(X_data, y_val))
    return skf_list

def get_train_test_data(fold_idx, skf_list, X_data, y_val):
    train_index, test_index = skf_list[fold_idx]
    X_train = [X_data[train_index, 0], X_data[train_index, 1]]
    X_test = [X_data[test_index, 0], X_data[test_index, 1]]
    y_train = y_val[train_index]
    y_test = y_val[test_index]
    return [X_train, y_train, X_test, y_test]

def sample_parameters(k, l_p, batch_size, epochs, c, n_params):
    params_dict = dict(k = k, l_p = l_p, batch_size = batch_size, epochs = epochs, c = c)
    rng = np.random.RandomState(10)
    param_list = list(ParameterSampler(params_dict, n_iter = n_params, random_state = rng))
    return param_list

def cv_scan(m, X_train, y_train, X_test, y_test, param_list, model_type, fold_idx, folder, out):
    """ 5-fold cross validation with parameters scanning """
    for param in param_list:
        if model_type == 'symLMF':
            mf_model = symLMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])
        elif model_type == 'symPMF':
            mf_model = symPMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])
        else:
            mf_model = symNMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])

        model, _, _ = mf_model.train_model(m, X_train, y_train, X_test, y_test)
        test_pred = mf_model.get_test_prediction(model, X_test)
        aucScore = roc_auc_score(y_test, test_pred)
        # aupr calculation
        precision, recall, thresholds = precision_recall_curve(y_test, test_pred)
        aupr = auc(recall, precision)
        outline = [param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'], round(aucScore, 4), round(aupr, 4)]
        out.write('\t'.join([str(x) for x in outline]) + '\n')
        predict_file = folder + str(param['k']) + '_' + str(param['l_p']) + '_' + str(param['c']) + '_' + str(fold_idx) + '_test_prediction.txt'
        save_pred_file(X_test, y_test, test_pred, predict_file)

def save_pred_file(X_test, y_test, test_pred, predict_file):
    pros_A, pros_B = X_test[0], X_test[1]
    df_predict = pd.DataFrame(pros_A, columns = ['Protein_A'])
    df_predict['Protein_B'] = pros_B
    df_predict['Label'] = y_test
    df_predict['Predicted_Score'] = test_pred
    df_predict.to_csv(predict_file, header = True, index = False, sep = '\t')

def concat_files(file1, file2, col_names, target_name):
    df2 = pd.read_csv(file2, sep = '\t', names = col_names + [target_name])
    if os.path.isfile(file1):
        df1 = pd.read_csv(file1, sep = '\t')
        df = pd.concat([df1, df2], ignore_index = True)
    else:
        df = df2
    df.to_csv(file1, sep='\t', index=False)
    return df

def save_mean_file(df, group_key, avg, outfile):
    df_mean = df.groupby(group_key, as_index=False)[avg].mean()
    df_mean.sort_values(by=avg, axis=0, ascending=False, inplace=True)
    df_mean.to_csv(outfile, sep='\t', index=False)

def cv2_evaluate(m, X_train, y_train, X_test, y_test, param, model_type, model_output_file, top_n, top_pair_out_file):
    if os.path.isfile(model_output_file):
        model = keras.models.load_model(model_output_file)
    else:
        if model_type == 'symLMF':
            mf_model = symLMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])
        elif model_type == 'symPMF':
            mf_model = symPMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])
        else:
            mf_model = symNMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])

        model, _, _ = mf_model.train_model(m, X_train, y_train, X_test, y_test)
        model.save(model_output_file)

    a2b_train = {}
    for i in range(y_train.shape[0]):
        a, b = X_train[0][i], X_train[1][i]
        a2b_train.setdefault(a, set()).add(b)
        a2b_train.setdefault(b, set()).add(a)

    a2b_test_pos = {}
    for i in range(y_test.shape[0]):
        if y_test[i] == 0:
            continue
        a, b = X_test[0][i], X_test[1][i]
        if a not in a2b_train or b not in a2b_train:
            continue
        a2b_test_pos.setdefault(a, set()).add(b)
        a2b_test_pos.setdefault(b, set()).add(a)

    n_users = len(a2b_test_pos)
    prec_recall_n = np.zeros((n_users, 2))

    if os.path.isfile(top_pair_out_file):
        df = pd.read_csv(top_pair_out_file, sep = '\t', usecols = ['Protein_A', 'Label'])
        df_grouped = df.groupby('Protein_A')['Label'].sum().reset_index()
        a2tp = df_grouped.set_index('Protein_A')['Label'].to_dict()
        j = 0
        for a, tp in a2tp.items():
            P_tot = len(a2b_test_pos[a])
            prec_recall_n[j, 0] = tp/P_tot
            prec_recall_n[j, 1] = tp/top_n
            j += 1
    else:
        j = 0
        top_pairs = np.zeros((n_users * top_n, 4))
        for a, test_pos_set in a2b_test_pos.items():
            pro_b = []
            label = []

            for i in range(m):
                if i == a or i in a2b_train[a]:
                    continue
                pro_b.append(i)
                if i in test_pos_set:
                    label.append(1)
                else:
                    label.append(0)

            pro_a = [a] * len(pro_b)
            pro_a = np.array(pro_a)
            pro_b = np.array(pro_b)
            label = np.array(label)
            y_pred = model.predict([pro_a, pro_b])
            y_pred = y_pred.flatten()
            top_idx = np.argpartition(y_pred, -top_n)[-top_n: ]
            TP = sum(label[top_idx])
            P_tot = len(test_pos_set)
            prec_recall_n[j, 0] = TP/top_n
            prec_recall_n[j, 1] = TP/P_tot

            top_pairs[top_n*j: top_n*(j+1), 0] = pro_a[top_idx]
            top_pairs[top_n*j: top_n*(j+1), 1] = pro_b[top_idx]
            top_pairs[top_n*j: top_n*(j+1), 2] = label[top_idx]
            top_pairs[top_n*j: top_n*(j+1), 3] = y_pred[top_idx]

            j += 1

        df = pd.DataFrame(top_pairs, columns = ['Protein_A', 'Protein_B', 'Label', 'Predicted_Score'])
        df = df.astype({'Protein_A': int, 'Protein_B': int, 'Label': int})
        df.to_csv(top_pair_out_file, sep = '\t', header = True, index = False)

    mean_prec_recall_n = np.mean(prec_recall_n, axis=0)
    return mean_prec_recall_n

def train_save_model(m, X_train, y_train, X_test, y_test, param, model_type, model_out_file, embedding_out_file, bias_out_file):

    if model_type == 'symLMF':
        mf_model = symLMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])
    elif model_type == 'symPMF':
        mf_model = symPMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])
    else:
        mf_model = symNMF(param['k'], param['l_p'], param['batch_size'], param['epochs'], param['c'])

    model, user_embedding_model, user_bias_model = mf_model.train_model(m, X_train, y_train, X_test, y_test)
    model.save(model_out_file)

    user_embedding_matrix = mf_model.extract_embedding_matrix(user_embedding_model, m)
    user_bias_matrix = mf_model.extract_embedding_matrix(user_bias_model, m)

    np.save(embedding_out_file, user_embedding_matrix)
    np.save(bias_out_file, user_bias_matrix)

def get_prediction_matrix(embedding_matrix, bias_matrix):
    m = embedding_matrix.shape[0]
    prediction_matrix = np.dot(embedding_matrix, embedding_matrix.T)
    prediction_matrix += np.repeat(bias_matrix, m, axis=1)
    prediction_matrix += np.repeat(bias_matrix.T, m, axis=0)
    prediction_matrix = 1/(1+np.exp(-prediction_matrix))

    return prediction_matrix

def hide_training_data(prediction_matrix, X_data, y_val):
    for i in range(X_data.shape[0]):
        if y_val[i] == 0:
            continue
        prediction_matrix[X_data[i, 0], X_data[i, 1]] = 0

    for i in range(prediction_matrix.shape[0]):
        prediction_matrix[i, i] = 0

def get_similarity_matrix(embedding_matrix, bias_matrix):

    return cosine_similarity(np.concatenate((embedding_matrix, bias_matrix), axis = 1))

def get_top_vals(input_matrix, top_n):
    top_idx = input_matrix.argsort(axis = 1)[:, -top_n:]
    m = input_matrix.shape[0]
    top_matrix = np.zeros((m * top_n, 3))
    for i in range(m):
        top_val = input_matrix[i, top_idx[i]]
        sorted_idx = top_val.argsort()[::-1]
        updated_idx = top_idx[i, sorted_idx]
        updated_val = top_val[sorted_idx]

        top_matrix[i*top_n: (i+1)*top_n, 0] = i
        top_matrix[i*top_n: (i+1)*top_n, 1] = updated_idx
        top_matrix[i*top_n: (i+1)*top_n, 2] = updated_val

    return top_matrix

def get_top_pred_tot(prediction_matrix, top_n, cutoff):
    matrix_lower = np.tril(prediction_matrix, -1)  # use the lower diagonal only
    top_pred_ind = np.where(matrix_lower > cutoff) # select a subset to sort, save time
    top_pred = matrix_lower[top_pred_ind[0], top_pred_ind[1]]
    tmp_ind = np.argpartition(top_pred, -top_n)[-top_n:]
    top_pred_pairs = []
    for k in range(top_n):
        ind = tmp_ind[k]
        top_pred_pairs.append([top_pred_ind[0][ind], top_pred_ind[1][ind], top_pred[ind]])
    top_pred_pairs.sort(key = lambda x: x[2], reverse = True)
    return top_pred_pairs

def get_top_predictions(top_n, n_folds, X_data, y_val, folder):
    top_prediction_file = folder + 'top_' + str(top_n) + '_predictions.txt' # top n predictions for each item
    top_pred_tot_file = folder + 'top1000_predictions.txt' # top 1000 predictions overall
    top_similarity_file = folder + 'top_' + str(top_n) + '_similar_proteins.txt'

    i = 0  # first fold

    embedding_file = folder + 'fold' + str(i) + '_embedding.npy'
    bias_file = folder + 'fold' + str(i) + '_bias.npy'
    embedding_matrix = np.load(embedding_file, allow_pickle=True)
    bias_matrix = np.load(bias_file, allow_pickle=True)
    # check the matrix shape
    print('embedding shape: ' + str(embedding_matrix.shape))
    print('bias shape: ' + str(bias_matrix.shape))

    prediction_matrix = get_prediction_matrix(embedding_matrix, bias_matrix)
    print('prediction shape: ' + str(prediction_matrix.shape))

    similarity_matrix = get_similarity_matrix(embedding_matrix, bias_matrix)
    print('similarity shape: ' + str(similarity_matrix.shape))

    for i in range(1, n_folds):
        embedding_file = folder + 'fold' + str(i) + '_embedding.npy'
        bias_file = folder + 'fold' + str(i) + '_bias.npy'
        embedding_matrix = np.load(embedding_file)
        bias_matrix = np.load(bias_file)

        prediction_matrix += get_prediction_matrix(embedding_matrix, bias_matrix)
        prediction_matrix /= 2 # get average predictions

        similarity_matrix += get_similarity_matrix(embedding_matrix, bias_matrix)
        similarity_matrix /= 2

    hide_training_data(prediction_matrix, X_data, y_val)
    # get top 1000 predictions
    top_predicted_pairs_tot = get_top_pred_tot(prediction_matrix, 1000, 0.7)
    top_predicted_pairs_tot = np.array(top_predicted_pairs_tot)
    df_tot = pd.DataFrame(top_predicted_pairs_tot, columns=['Protein_A', 'Protein_B', 'Predicted_Score'])
    df_tot = df_tot.astype({'Protein_A': int, 'Protein_B': int})
    df_tot.to_csv(top_pred_tot_file, sep='\t', header=True, index=False)

    top_prediction_matrix = get_top_vals(prediction_matrix, top_n)
    print('top prediction shape: ' + str(top_prediction_matrix.shape))
    df = pd.DataFrame(top_prediction_matrix, columns=['Protein_A', 'Protein_B', 'Predicted_Score'])
    df = df.astype({'Protein_A': int, 'Protein_B': int})
    df.to_csv(top_prediction_file, sep='\t', header=True, index=False)
    
    for i in range(similarity_matrix.shape[0]):
        similarity_matrix[i, i] = 0 # exclude the protein itself from the top list
    
    top_similarity_matrix = get_top_vals(similarity_matrix, top_n)
    print('top similarity shape: ' + str(top_similarity_matrix.shape))
    df = pd.DataFrame(top_similarity_matrix, columns=['Protein_A', 'Protein_B', 'Similarity'])
    df = df.astype({'Protein_A': int, 'Protein_B': int})
    df.to_csv(top_similarity_file, sep='\t', header=True, index=False)

def get_top_pred_pairs(top_n, i, repeats, folder, X_data, y_val):
    j = 0  # first run

    embedding_file = folder + 'run' + str(i) + '_' + str(j) + '_embedding.npy'
    bias_file = folder + 'run' + str(i) + '_' + str(j) + '_bias.npy'
    embedding_matrix = np.load(embedding_file, allow_pickle=True)
    bias_matrix = np.load(bias_file, allow_pickle=True)
    # check the matrix shape
    print('embedding shape: ' + str(embedding_matrix.shape))
    print('bias shape: ' + str(bias_matrix.shape))

    prediction_matrix = get_prediction_matrix(embedding_matrix, bias_matrix)
    print('prediction shape: ' + str(prediction_matrix.shape))

    for j in range(1, repeats):
        embedding_file = folder + 'run' + str(i) + '_' + str(j) + '_embedding.npy'
        bias_file = folder + 'run' + str(i) + '_' + str(j) + '_bias.npy'
        embedding_matrix = np.load(embedding_file)
        bias_matrix = np.load(bias_file)

        prediction_matrix += get_prediction_matrix(embedding_matrix, bias_matrix)
        prediction_matrix /= 2  # get average predictions

    hide_training_data(prediction_matrix, X_data, y_val)

    return get_top_pred_tot(prediction_matrix, top_n, 0.9)

def get_top_random_pairs(top_n, repeats, n_pros, X_data, y_val):
    """
    top pairs with a random predictor
    """
    j = 0  # first run

    prediction_matrix = np.random.rand(n_pros, n_pros)

    for j in range(1, repeats):
        prediction_matrix += np.random.rand(n_pros, n_pros)
        prediction_matrix /= 2  # get average predictions

    hide_training_data(prediction_matrix, X_data, y_val)

    return get_top_pred_tot(prediction_matrix, top_n, 0.7)




























