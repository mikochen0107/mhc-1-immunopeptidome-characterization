import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class PeptideDataset:
    ROOT = '../datasets/MHC_I_el_allele_specific'.format(__file__)  # root directory containing peptide binding data
    ALL_AA = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    NUM_AA = len(ALL_AA)  # number of amino acids (20)
    
    def __init__(self, hla_allele, root=None, encoding='default', max_aa_len=14, padding='end', test_set='c004', input_format='linear'):
        '''
        Initialize dataset class for each human leukocyte antigen (HLA or MHC) allele.
        
        Parameters
        ----------
        hla_allele: str
            Folder name of HLA allele of interest
        
        root: str, optional
            Location of dataset
            
        encoding: str, optional
            Amino acid encoding style. Options: 'default', TBD
            
        max_aa_len: int, optional
            
        padding: str, optional
            Padding for amino acid sequence. Options: 'begin', 'end', 'after2', TBD
            
        test_set: str, optional
            Specify test set which should not be touched during model development.
            Options: 'c000', 'c001', 'c002', 'c003', 'c004'
        
        input_format: str
            Specify datum shape. Options: 'linear', '2d'
        '''
        
        self.hla_allele = hla_allele
        self.root = self.ROOT if root == None else root
        self.encoding = encoding
        self.max_aa_len = max_aa_len
        self.padding = padding
        self.test_set = test_set
        self.input_format = input_format
        
        self.data, self.targets, self.raw_data = self.parse_csv()
        
       
    def parse_csv(self):
        '''
        Open up CSV and gets pandas dataframe for initializing class' properties.
        
        Returns
        ----------
        data: dict of ndarray
            Dataset (N_i x M) for each peptide group
        
        targets: dict of ndarray
            Labels (N_i x 1) for each peptide group
        '''
        
        files = os.listdir(os.path.join(self.root, self.hla_allele))
        files.remove(self.test_set)  # remove test set
        
        raw_data = {}
        targets = {}
        
        for file in files:
            content = np.loadtxt(os.path.join(self.root, self.hla_allele, file), dtype='str')
            raw_data[file] = content[:, 0].astype('str')
            targets[file] = content[:, 1].astype(float)
        
        data = {}
        for file in files:
            data[file] = [] # initialize empty array for dict value
            for aa_seq in raw_data[file]:
                data[file].append(self.format_seq(aa_seq))
            data[file] = np.stack(data[file], axis=0)
        
        return data, targets, raw_data
    
    def format_seq(self, seq):
        '''
        Converts an amino acid string sequence into a binary padded format.
        
        Parameters
        ----------
        seq: str
            Sequence of amino acids, small or big letters
        
        Returns
        ----------
        feat_vect: ndarray
            Flat vector (N x 1) or 2D tensor ((N / BITS) x BITS) encoding amino acid sequence
        '''
        
        converted_seq, bits = self.encode_seq(seq)
        padded_converted_seq = self.pad(converted_seq, bits) 
        
        feat_vect = np.fromstring(' '.join(padded_converted_seq), sep=' ', dtype=float)  # convert to binary ndarray
        if self.input_format == '2d':
            feat_vect.reshape((int(len(padded_converted_seq) / bits), bits))
        
        return feat_vect
    
    def encode_seq(self, _seq):
        '''
        Converts a string into an linear binary string of features.
        
        Parameters
        ----------
        _seq: str
            Sequence of amino acids, small or big letters
        
        Returns
        ----------
        encoded_seq: ndarray
            String of binaries encoding each amino acid (N x 1)
        bits: int
            Number of bits used to encode each amino acid (default: NUM_AA=20) 
        '''
        
        seq = _seq.upper()  # make amino acid sequence all CAPS
        encoded_seq = ''
        for aa in seq:
            if self.encoding == 'default':
                encoded_aa = self._encode_default(aa)
                bits = len(encoded_aa)  # get length of binary code for each amino acid
            encoded_seq += encoded_aa  # append amino acid binary to the sequence
            
        return encoded_seq, bits
            
    def _encode_default(self, aa):
        '''
        Converts a string character into an encoded linear binary string with the default encoding
        
        Parameters
        ----------
        aa: str
            String of length 1 representing one amino acid
        
        Returns
        ----------
        bin_aa: str
            Binary string encoding an amino acid
        '''
    
        bin_placeholder = '0' * self.NUM_AA
        insert_pos = self.ALL_AA.index(aa)
        bin_aa = bin_placeholder[:insert_pos] + '1' + bin_placeholder[insert_pos+1:]
        
        return bin_aa
    
    def pad(self, seq, bits):
        '''
        Pad binary string sequence to unify the sequence length.
        
        Parameters
        ----------
        seq: string
            Binary sequence of amino acids
        
        Returns
        ----------
        padded_seq: string
            String of padded amino acid sequence (N x 1)
        '''
        
        pad_len = self.max_aa_len * bits - len(seq)  # number of bits to pad
        pad_bits = '0' * pad_len
        
        if self.padding == 'begin':
            padded_seq = pad_bits + seq
        elif self.padding == 'end':
            padded_seq = seq + pad_bits
        elif self.padding == 'after2':
            pos = 2 * bits
            padded_seq = seq[:pos] + pad_bits + seq[pos:]
            
        return padded_seq
    
    def get_fold(self, fold_idx=[0, 1, 2, 3], randomize=True, raw_data=False):
        '''
        Extracts the desired folds, concatenates them into a single list, and randomizes the data points
        
        Parameters
        ----------
        fold_idx: list
            List of number (from 0 to 4) to specify folds
        
        Returns
        ----------
        features: ndarray
            Concatenated list of the desired folds
        '''
        
        which_data = self.data if raw_data else self.raw_data
        _data_fold = np.vstack([which_data['c00{}'.format(f)] for f in fold_idx])
        if randomize:
            data_fold = _data_fold[np.random.permutation(_data_fold.shape[0])]
        
        return data_fold
    
    def basic_dataviz(self, fold_idx=[0, 1, 2, 3]):
        '''
        Visualizes simple statistics 
        
        Parameters
        ----------
        fold_idx: list
            List of number (from 0 to 4) to specify folds
        
        Returns
        ----------
        features: plt.figure
            Histogram of peptide lengths,
            amino acid distribution,
            sizes of folds
        '''
        
        # get the folds of interest 
        data = self.get_fold(fold_idx, raw_data=True)
        
        # histogram of peptide lengths
        pep_len = []
        for i in range(len(data)):
            length = len(data[i])
            pep_len.append(length)
        
        plt.figure(figsize=(12,6))
        plt.hist(pep_len, bins = np.arange(min(pep_len)-0.5, max(pep_len)+1.5 , 1.0))
        plt.show()
        
        # amino acid distribution
        pep_count = Counter()
        for pep in data:
            pep_count += Counter(pep)
            
        plt.figure(figsize=(12,6))
        plt.bar(pep_count.keys(), pep_count.values())
        plt.show()
        
        # size of folds
        fold_size = []
        for i in fold_idx:
            fold_size.append(len(data[i])) 
        
        plt.figure(figsize=(12,6))
        plt.bar(fold_idx, fold_size)
        plt.xticks(np.arange(0, 5))
        plt.show()