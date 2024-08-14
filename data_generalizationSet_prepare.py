import os
from sklearn.model_selection import KFold
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch_geometric import data as DATA
from rdkit.Chem import ChemicalFeatures
import networkx as nx
import os.path as osp
from rdkit import RDConfig
import torch
from torch_geometric.data import InMemoryDataset
from rdkit.Chem import AllChem
from data_pubchemFP import GetPubChemFPs

vocab_drug= {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def smiles_to_int(smiles):
    return [vocab_drug[s] for s in smiles]

def GetMolFingerprints(smile, nBits=2048):
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    FP = fp.ToBitString()
    FP_array = []
    for i in range(len(FP)):
        FP_value = float(FP[i])
        FP_array.append(FP_value)
    # FP_array = np.array(FP_array)  # 将列表转换为NumPy数组
    return FP_array

vocab_protein = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
				"U": 19, "T": 20, "W": 21,
				"V": 22, "Y": 23, "X": 24,
				"Z": 25 }

def seqs_to_int(target):
    return [vocab_protein[s] for s in target]

def get_protein_kmer_featue(input):
    strr = input
    list=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    data0 = np.zeros(20)
    n = len(strr)
    for m in range(n):
        for i in range(20):
            if strr[m] == list[i]:
                data0[i] += 1
                break
    data0 = data0.reshape(-1, )
    data0 = (data0 - np.mean(data0)) / np.std(data0)

    data1 = np.zeros((20, 20))
    n = len(strr)
    ct = 0
    for m in range(n - 1):
        ct = 0
        for i in range(20):
            if strr[m] == list[i]:
                for j in range(20):
                    if strr[m+1] == list[j]:
                        data1[i][j] += 1
                        ct += 1
                        break
            if ct==1:
                break
    data1 = data1.reshape(-1, )
    data1 = (data1 - np.mean(data1)) / np.std(data1)
    data2 = np.zeros((20, 20, 20))
    n = len(strr)
    for m in range(n - 2):
        ct = 0
        for i in range(20):
            if strr[m] == list[i]:
                for j in range(20):
                    if strr[m+1] == list[j]:
                        for k in range(20):
                            if strr[m+2] == list[k]:
                                data2[i][j][k] += 1
                                ct += 1
                                break
                    if ct == 1:
                        break
            if ct==1:
                break
    data2 = data2.reshape(-1, )
    data2 = (data2 - np.mean(data2)) / np.std(data2)
    data3 = np.concatenate((data0, data1, data2))
    return data3


class GNNDataset(InMemoryDataset):

    def __init__(self, root, train=False, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train==False:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass


    def process_data(self, data_path, graph_dict):
        df = pd.read_csv(data_path)

        data_list = []
        for i, row in df.iterrows():
            smiles = row['Smiles']
            sequence = row['Sequence']
            label = row['Affinity']

            if graph_dict[smiles] is None:
                continue

            x, edge_index, edge_attr = graph_dict[smiles]

            '''data normalization'''
            x = (x - x.min()) / (x.max() - x.min())

            '''Generate molecular morgen fingerprints'''
            fingerprint = GetMolFingerprints(smiles, 2048)

            '''MACCS、PubChem、Pharmacophore ErG'''
            mixed_fingerprint = []
            mol = Chem.MolFromSmiles(smiles)
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            fp_pubcfp = GetPubChemFPs(mol)
            mixed_fingerprint.extend(fp_maccs)
            mixed_fingerprint.extend(fp_phaErGfp)
            mixed_fingerprint.extend(fp_pubcfp)

            '''Convert drug smiles to integers'''
            smiles_int = smiles_to_int(smiles)
            smiles_len = 200
            if len(smiles_int) < smiles_len:
                smiles_int = np.pad(smiles_int, (0, smiles_len - len(smiles_int)))
            else:
                smiles_int = smiles_int[:smiles_len]


            '''Convert protein sequence to integers'''
            target = seqs_to_int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            '''protein k-mer feature'''
            kmer = get_protein_kmer_featue(sequence)

            # Get Labels
            try:
                x= np.array(x)
                edge_index = np.array(edge_index)
                edge_attr = np.array(edge_attr)

                data = DATA.Data(
                    x=torch.tensor(x,dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                    y=torch.tensor([label], dtype=torch.float32),
                    target=torch.tensor([target], dtype=torch.long),
                    smiles=torch.tensor([smiles_int], dtype=torch.long),
                    kmer=torch.tensor([kmer],dtype=torch.float32),
                    mixedFP=torch.tensor([mixed_fingerprint], dtype=torch.float32)
                )
            except:
                    print("unable to process: ", smiles)

            data_list.append(data)

        return data_list

    def process(self):
        df_test = pd.read_csv(self.raw_paths[0])

        smiles = df_test['Smiles'].unique()
        graph_dict = dict()
        print("converting molecular into graph structures...")
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            graph = self.mol_to_graph(mol)
            graph_dict[smile] = graph

        test_list = self.process_data(self.raw_paths[0], graph_dict)

        if self.pre_filter is not None:
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            test_list = [self.pre_transform(test) for test in test_list]

        print('Graph construction done. Saving to file.')

        data, slices = self.collate(test_list)
        # save preprocessed test data.csv:
        torch.save((data, slices), self.processed_paths[0])

    def get_nodes(self, graph):
        feat = []
        for n, d in graph.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]  # atom type (one-hot)
            h_t.append(d['a_num'])  # atomic num(integer)
            h_t.append(d['acceptor'])  # accepts electrons(binary)
            h_t.append(d['donor'])  # accepts electrons(binary)
            h_t.append(int(d['aromatic']))  # aromatic(binary)芳香族
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]  # Hybridization [sp,sp2,sp3] (one-hot)
            h_t.append(d['num_h'])  # number of connected hydrogen(integer)
            # 5 more
            h_t.append(d['ExplicitValence'])  # explicit valence of the atom(integer)
            h_t.append(d['FormalCharge'])  # formal charge of the atom(integer)
            h_t.append(d['ImplicitValence'])  # implicit valence of the atom(integer)
            h_t.append(d['NumExplicitHs'])  # number of implicit H the atom is bound to(integer)
            h_t.append(d['NumRadicalElectrons'])  # number of radical electrons for the atom(integer)
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])

        return node_attr

    def get_edges(self, graph):
        e = {}
        for n1, n2, d in graph.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t

        if len(e) == 0:
            return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol_to_graph(self, mol):
        if mol is None:
            return None
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')  # RDConfig.RDDataDir：RDKit library where data files are stored
        chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        graph = nx.DiGraph()  # create an empty digraph object graph

        # Create nodes
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            graph.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    graph.nodes[n]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for n in node_list:
                    graph.nodes[n]['acceptor'] = 1

        # README Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    graph.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(graph)
        edge_index, edge_attr = self.get_edges(graph)

        return node_attr, edge_index, edge_attr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dataSetName', choices=["EC50", "IC50", "Ki", "Kd"],
                        help='Enter which dataset to use for the experiment')
    args = parser.parse_args()
    dataset = args.dataSetName

    generalization_set_list = ['ER','GPCR','IonChannel','Kinase']

    for generalization_set in generalization_set_list:
        '''load dataset from text file'''
        print(f"Loading generalization_set {generalization_set} data...")

        '''reading data'''
        input_path = (f'./dataset/{dataset}/GeneralizationSet_{generalization_set}/raw/')
        data = pd.read_csv(input_path + 'data_test.csv')

        print(f"Generating generalization_set {generalization_set} graph data...")
        GNNDataset(f'dataset/{dataset}/GeneralizationSet_{generalization_set}')
        print(f"Generating generalization_set {generalization_set} graph data Finished!!!")
        print()
        print()
