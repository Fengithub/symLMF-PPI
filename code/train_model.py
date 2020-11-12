import numpy as np
import os
from helper_functions import *
from multiprocessing import Pool

if __name__ == '__main__':
    binary = True
    data_source = 'tissue'
    model_type = 'symLMF'
    repeats = 20
    top_n = 100

    param = {'epochs': 20, 'batch_size': 64, 'k': 100, 'l_p': 1e-6, 'c': 1}
    folder = 'final_model/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    data_file = 'pros_AB.txt'
    n_pros, X_data, y_train = parse_data(data_file, binary, data_source)
    X_train = [X_data[:, 0], X_data[:, 1]] # train on the whole dataset
    X_test = [X_data[-100:, 0], X_data[-100:, 1]] # just use as input for the model function, don't really need it
    y_test = y_train[-100: ]
    inputs = []
    for i in range(repeats):
        model_out_file = folder + 'fold' + str(i) + '_model'
        np.random.seed(2020 + i*2)
        if not os.path.exists(model_out_file):
            embedding_out_file = folder + 'fold' + str(i) + '_embedding'
            bias_out_file = folder + 'fold' + str(i) + '_bias'
            inputs.append((n_pros, X_train, y_train, X_test, y_test, param, model_type, model_out_file, embedding_out_file, bias_out_file))

    if inputs:
        p = Pool(processes = len(inputs))
        p.starmap(train_save_model, inputs)
        p.close()

    get_top_predictions(top_n, repeats, X_data, y_train, folder)
