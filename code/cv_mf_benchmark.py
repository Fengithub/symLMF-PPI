import numpy as np
import os
from helper_functions import *

def run_cv_scan():
    binary = True
    orgs = ['human', 'yeast']
    data_source = 'benchmark'
    model_type = 'symNMF'
    n_folds = 5

    epochs = [20, 30]
    batch_size = [64]
    k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    l_p = [1e-6, 1e-4, 1e-2]
    c = [1, 2, 3, 4]
    n_params = 20
    param_list = sample_parameters(k, l_p, batch_size, epochs, c, n_params)

    for org in orgs:
        folder = '../' + org + '/' + model_type + '_result/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        data_file = '../' + org + '/pros_AB.txt'
        n_pros, X_data, y_val = parse_data(data_file, binary, data_source)
        skf_list = split_data(n_folds, X_data, y_val)
        cur_auc_file = folder + 'CV_scan_auc_cur.txt'
        out = open(cur_auc_file, 'w')
        for fold_idx in [0, 1，2, 3，4]:
            X_train, y_train, X_test, y_test = get_train_test_data(fold_idx, skf_list, X_data, y_val)
            cv_scan(n_pros, X_train, y_train, X_test, y_test, param_list, model_type, fold_idx, folder, out)
        out.close()

        all_auc_file = folder + 'CV_scan_auc.txt'
        mean_file = all_auc_file[:-4] + '_mean.txt'
        col_names = ['k', 'l', 'batch_size', 'epochs', 'c']
        target_name = 'auc'
        df = concat_files(all_auc_file, cur_auc_file, col_names, target_name)
        save_mean_file(df, col_names, target_name, mean_file)

if __name__ == '__main__':
    run_cv_scan()



