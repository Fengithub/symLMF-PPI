import numpy as np
import os
from sklearn.model_selection import train_test_split
from helper_functions import *
from multiprocessing import Pool

def get_top_counts(top_n, top_pred_pairs, test_pos_pairs, top_counts, i):
    count = 0
    for k in range(top_n):
        a, b = top_pred_pairs[k][0], top_pred_pairs[k][1]
        if (a, b) in test_pos_pairs or (b, a) in test_pos_pairs:
            count += 1
        top_counts[k, i] = count

if __name__ == '__main__':
    binary = True
    data_source = 'tissue'
    model_type = 'symLMF'
    split_times = 5 # number of independent random split runs
    repeats = 10 # number of repeated runs within each split
    top_n = 1000
    hide = 0.5 # proportion of hidden examples

    param = param = {'epochs': 20, 'batch_size': 64, 'k': 90, 'l_p': 1e-4, 'c': 2}
    folder = 'hide_model/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    data_file = 'pros_AB.txt'
    n_pros, X_data, y = parse_data(data_file, binary, data_source)
    inputs = []
    for i in range(split_times):
        X_train_tmp, X_test_tmp, y_train, y_test = train_test_split(X_data, y, test_size=hide, random_state=i*10)
        X_train = [X_train_tmp[:, 0], X_train_tmp[:, 1]]
        X_test = [X_test_tmp[:, 0], X_test_tmp[:, 1]]
        for j in range(repeats):
            model_out_file = folder + 'run' + str(i) + '_' + str(j) + '_model'
            np.random.seed(2020 + j*5)
            if not os.path.exists(model_out_file):
                embedding_out_file = folder + 'run' + str(i) + '_' + str(j) + '_embedding'
                bias_out_file = folder + 'run' + str(i) + '_' + str(j) + '_bias'
                inputs.append((n_pros, X_train, y_train, X_test, y_test, param, model_type, model_out_file, embedding_out_file, bias_out_file))

    if inputs:
        p = Pool(processes = len(inputs))
        p.starmap(train_save_model, inputs)
        p.close()

    top_counts_file = folder + 'top' + str(top_n) + '_counts.npy'
    top_random_counts_file = folder + 'top' +  str(top_n) + '_random_counts.npy'

    top_counts = np.zeros((top_n, split_times))
    top_random_counts = np.zeros((top_n, split_times))
    for i in range(split_times):
        X_train_tmp, X_test_tmp, y_train, y_test = train_test_split(X_data, y, test_size=hide, random_state=i*10)
        test_pos_pairs = set()
        for ind in range(y_test.shape[0]):
            if y_test[ind] > 0:
                test_pos_pairs.add((X_test_tmp[ind, 0], X_test_tmp[ind, 1]))

        top_pred_pairs = get_top_pred_pairs(top_n, i, repeats, folder, X_train_tmp, y_train)
        get_top_counts(top_n, top_pred_pairs, test_pos_pairs, top_counts, i)

        top_random_pairs = get_top_random_pairs(top_n, repeats, n_pros, X_train_tmp, y_train)
        get_top_counts(top_n, top_random_pairs, test_pos_pairs, top_random_counts, i)

    np.save(top_counts_file, top_counts)
    np.save(top_random_counts_file, top_random_counts)





