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

def evaluate_top_n():
    binary = True
    org = 'human'
    data_source = 'benchmark'
    model_type = 'symLMF'
    n_folds = 5
    top_n = 10

    param = {'epochs': 20, 'batch_size': 64, 'k': 40, 'l_p': 1e-4, 'c': 1}

    folder = '../' + org + '/' + model_type + '_result/selected/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    data_file = '../' + org + '/pros_AB.txt'
    outfile = folder + 'mean_prec_recall@' + str(top_n) + '.txt'
    n_pros, X_data, y_val = parse_data(data_file, binary, data_source)
    skf_list = split_data(n_folds, X_data, y_val)
    mean_prec_recall_res = np.zeros((n_folds, 2))
    for fold_idx in range(n_folds):
        model_output_file = folder + 'fold' + str(fold_idx) + '_model'
        top_pair_out_file = folder + 'fold' + str(fold_idx) + '_top' + str(top_n) + '.txt'
        X_train, y_train, X_test, y_test = get_train_test_data(fold_idx, skf_list, X_data, y_val)
        mean_prec_recall_res[fold_idx, :] = cv2_evaluate(n_pros, X_train, y_train, X_test, y_test, param, model_type, model_output_file, top_n, top_pair_out_file)

    mean_prec_recall = np.mean(mean_prec_recall_res, axis = 0)
    std_prec_recall = np.std(mean_prec_recall_res, axis = 0)

    outline = [[mean_prec_recall[0], std_prec_recall[0]], [mean_prec_recall[1], std_prec_recall[1]]]
    outline = np.array(outline)
    np.savetxt(outfile, outline, fmt = '%.4f')

if __name__ == '__main__':
    # run_cv_scan()
    evaluate_top_n()



