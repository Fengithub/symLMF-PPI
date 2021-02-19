"""
Reference:Yao, Y.; Du, X.; Diao, Y.; Zhu, H., An integration of deep learning with feature embedding 
for protein-protein interaction prediction. PeerJ 2019, 7, e7126.

code modified based on the originial code available on Github: https://github.com/xal2019/DeepFE-PPI
"""

import sentencepiece as spm
import numpy as np
import pandas as pd

def get_words(protein):
    sp = spm.SentencePieceProcessor()
    sp.Load('./seq2word_model/get_words_extended.model')
    sp_pro = sp.EncodeAsPieces(protein)

    return [sp_pro]

def parse_data(protein_file, pa_file, pb_file):
    pro_df = pd.read_csv(protein_file, sep = '\t')
    pa_df = pro_df['Fasta_A'].to_list()
    pb_df = pro_df['Fasta_B'].to_list()

    pro_pa = []
    pro_pb = []

    for i in range(len(pa_df)):
        pro_a = get_words(str(pa_df[i]))
        pro_pa.append(pro_a)

        pro_b = get_words(str(pb_df[i]))
        pro_pb.append(pro_b)

    pa = pd.DataFrame(pro_pa, columns = ['pa'])
    pb = pd.DataFrame(pro_pb, columns = ['pb'])

    pa['pre_words'] = pa.apply(lambda x: ''.join(x['pa'][1:-1].split(',')), axis = 1)
    pa['words'] = pa.apply(lambda x: x['pre_words'].replace("'",""), axis = 1)
    pa.drop(['pre_words'],axis=1, inplace=True)
    pa.drop(['pa'],axis=1, inplace=True)

    pb['pre_words'] = pb.apply(lambda x: ''.join(x['pb'][1:-1].split(',')), axis = 1)
    pb['words'] = pb.apply(lambda x: x['pre_words'].replace("'",""), axis = 1)
    pb.drop(['pre_words'],axis=1, inplace=True)
    pb.drop(['pb'],axis=1, inplace=True)

    pa.to_csv(pa_file, header = False, index = False, sep = '\t')
    pb.to_csv(pb_file, header = False, index = False, sep = '\t')

def get_data(dataset, org, pro_type):
    folder = folder = '../../datasets/' 
    pro_file = folder + dataset + '/' + org + '_' + pro_type + '_proteins.txt'

    pa_file = folder + dataset + '/' + org + '_' + pro_type + '_pa.txt'
    pb_file = folder + dataset + '/' + org + '_' + pro_type + '_pb.txt'

    parse_data(pro_file, pa_file, pb_file)

if __name__ == '__main__':

    for pro_type in ['positive', 'negative']:
        get_data('H.sapiens-extended', 'human', pro_type)    ## 'S.cerevisiae-extended', 'yeast' 
    
