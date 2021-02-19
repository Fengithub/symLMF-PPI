import pandas as pd 
import numpy as np 

def parse_data(data_file, fasta_file, pos_file, neg_file):
    data_df = pd.read_csv(data_file, sep = '\t')
    fasta_df = pd.read_csv(fasta_file, sep = '\t')
    unip2fasta = fasta_df.set_index('Uniprot_ID')['Fasta'].to_dict()
    data_df['Fasta_A'] = data_df.apply(lambda x: unip2fasta[x['Uniprot_A']], axis = 1)
    data_df['Fasta_B'] = data_df.apply(lambda x: unip2fasta[x['Uniprot_B']], axis = 1)

    data_df['Interaction'] = data_df.apply(lambda x: 1 if x['Interaction'] > 0 else 0, axis = 1)
    pos_df = data_df[data_df['Interaction'] > 0]
    neg_df = data_df[data_df['Interaction'] == 0]

    pos_df.to_csv(pos_file, index = False, sep = '\t')
    neg_df.to_csv(neg_file, index = False, sep = '\t')

def get_sequences(dataset, org):
    folder = '../datasets/' 
    data_file = folder + dataset + '/PPI_' + org + '_2019.txt'
    fasta_file = folder + dataset + '/unip2fasta_' + org + '.txt'

    pos_file = folder + dataset + '/' + org + '_positive_proteins.txt'
    neg_file = folder + dataset + '/' + org + '_negative_proteins.txt'

    parse_data(data_file, fasta_file, pos_file, neg_file)

if __name__ == '__main__':
    
    get_sequences('H.sapiens-extended', 'human')   ## 'S.cerevisiae-extended', 'yeast' 
