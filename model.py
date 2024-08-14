import torch

from module_drug import *
from module_protein import *
from module_attention import *

class DTANet(nn.Module):
    def __init__(self,hp, block_num, vocab_protein_size, vocab_drug_size, out_dim=1):
        super().__init__()
        self.protein_encoder = Protein_Seq_Representation(block_num, vocab_protein_size, hp.embedding_size,out_dim=hp.module_out_dim)
        self.kmer_encoder = Protein_Kmer_Representation(hp)


        self.drug_graph_encoder = Drug_Graph_Representation(num_input_features=22, out_dim=hp.module_out_dim, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

        self.drug_semantics_embedder = nn.Embedding(vocab_drug_size, hp.embedding_size)
        self.drug_semantics_encoder = Drug_Semantics_Representation(hp.embedding_size, hp.module_out_dim,name='drug_semantics')

        self.drug_morganFP_encoder = Drug_MorganFP_Representation(input_dim=2048,output_dim=hp.module_out_dim,hidden_dims_lst=[2048,1024,256])
        self.drug_mixedFP_encoder = Drug_MixedFP_Representation(hp)

        self.drug_attention = Drug_CrossAttentionBlock(hp.module_out_dim,n_heads=hp.attention_n_heads,dropout=0.1)
        self.protein_attention = Protein_CrossAttentionBlock(hp.module_out_dim, n_heads=hp.attention_n_heads,dropout=0.1)
        self.drug_protein_attention = Drug_Protein_CrossAttentionBlock(hp.module_out_dim, n_heads=hp.attention_n_heads,dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(hp.module_out_dim * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, data):
        '''protein feature extraction module'''
        # target: torch.Size([512, 900])
        target = data.target
        protein_sequence_feature = self.protein_encoder(target)

        '''protein k-mer feature extraction module'''
        kmer = data.kmer
        protein_kmer_feature = self.kmer_encoder(kmer)

        '''drug molecule feature extraction module'''
        drug_graph_feature = self.drug_graph_encoder(data)

        '''drug semantics feature extraction module'''
        semantics = data.smiles
        semantics_vectors = self.drug_semantics_embedder(semantics)
        drug_semantics_feature = self.drug_semantics_encoder(semantics_vectors)

        # '''drug morgan fingerprint feature extraction module'''
        # morganFP = data.morganFP
        # drug_morganFP_feature = self.drug_morganFP_encoder(morganFP)

        '''Drug mixed fingerprint (MACCS、PubChem、Pharmacophore ErG) feature extraction module'''
        mixedFP = data.mixedFP
        drug_mixedFP_feature = self.drug_mixedFP_encoder(mixedFP)


        drug_semantics_graph_feature = self.drug_attention(drug_semantics_feature, drug_graph_feature)
        # drug_graph_mixedFP_feature = self.drug_attention(drug_graph_feature, drug_mixedFP_feature)
        # protein_feature = self.protein_attention(protein_sequence_feature,protein_kmer_feature)

        drug_protein_feature = torch.cat([drug_semantics_graph_feature,drug_mixedFP_feature,protein_sequence_feature,protein_kmer_feature],dim=-1)

        # drug_protein_feature = self.drug_protein_attention(drug_feature,protein_feature)
        output = self.classifier(drug_protein_feature)

        return output


