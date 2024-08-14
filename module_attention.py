import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()


    def forward(self, query, key, value, mask=None):
        """
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Key and Value should always come from the same source (Aiming to forcus on), Query comes from the other source
        Self-Att : Both three Query, Key, Value come from the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix


'''drug attention module'''
class Drug_CrossAttentionBlock(nn.Module):

    def __init__(self,hidden_size,n_heads,dropout):
        super(Drug_CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hidden_size, n_heads, dropout)

    def forward(self, drug_feature1, drug_feature2):
        """
            :graph_feature : A batch of 1D tensor for representing the Graph information from compound
            :morgan_feature: A batch of 1D tensor for representing the ECFP information from compound
            :sequence_feature: A batch of 1D tensor for representing the information from protein sequence
        """

        '''cross attention for drug information enrichment'''
        drug_feature = self.att(drug_feature1, drug_feature2, drug_feature2)
        # drug_semantics_graph_mixedFP_feature = self.att(drug_semantics_graph_feature, drug_mixedFP_feature, drug_mixedFP_feature)

        # drug_semantics_graph_mixedFP_feature = self.att(drug_semantics_graph_mixedFP_feature,drug_semantics_graph_mixedFP_feature,drug_semantics_graph_mixedFP_feature)

        '''cross-attentcion for interaction'''
        # output = self.att(drug_semantics_graph_mixedFP_feature,protein_feature,protein_feature)

        return drug_feature


'''protein attention module'''
class Protein_CrossAttentionBlock(nn.Module):

    def __init__(self,hidden_size,n_heads,dropout):
        super(Protein_CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hidden_size, n_heads, dropout)

    def forward(self, protein_sequence_feature, protein_kmer_feature):
        """
            :graph_feature : A batch of 1D tensor for representing the Graph information from compound
            :morgan_feature: A batch of 1D tensor for representing the ECFP information from compound
            :sequence_feature: A batch of 1D tensor for representing the information from protein sequence
        """

        '''cross attention for protein information enrichment'''
        protein_feature = self.att(protein_kmer_feature, protein_sequence_feature, protein_sequence_feature)

        '''cross-attentcion for interaction'''
        # output = self.att(drug_semantics_graph_mixedFP_feature,protein_feature,protein_feature)

        return protein_feature


'''drug_protein attention module'''
class Drug_Protein_CrossAttentionBlock(nn.Module):

    def __init__(self,hidden_size,n_heads,dropout):
        super(Drug_Protein_CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hidden_size, n_heads, dropout)

    def forward(self, drug_feature, protein_feature):
        """
            :graph_feature : A batch of 1D tensor for representing the Graph information from compound
            :morgan_feature: A batch of 1D tensor for representing the ECFP information from compound
            :sequence_feature: A batch of 1D tensor for representing the information from protein sequence
        """

        '''cross attention for drug and protein information enrichment'''
        drug_protein_feature = self.att(drug_feature, protein_feature, protein_feature)

        '''cross-attentcion for interaction'''
        # output = self.att(drug_semantics_graph_mixedFP_feature,protein_feature,protein_feature)

        return drug_protein_feature
