#set transformer code
#shoutout Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye full citation at bottom
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
#archieture embedder hopefully helps converge quicker to better perplexity
class ArchitectureEmbedder(nn.Module):
    def __init__(self, hidden_dim, num_types=10, num_semantics=200): # Increased to 200
        super().__init__()
        self.hidden_dim = hidden_dim
        
        #Structural IDs
        self.type_emb = nn.Embedding(num_types, hidden_dim)
        
        # Semantic Role IDs (Dynamic Vocab)
        self.semantic_emb = nn.Embedding(num_semantics, hidden_dim)
        
        # Continuous Properties
        self.shape_proj = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.param_cnt_proj = nn.Linear(1, hidden_dim)
        
        
        self.out_proj = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, type_ids, semantic_ids, shape_vecs, param_counts):
        # dims
        if type_ids.dim() == 2: type_ids = type_ids.squeeze(-1)
        if semantic_ids.dim() == 2: semantic_ids = semantic_ids.squeeze(-1)
        if param_counts.dim() == 1: param_counts = param_counts.unsqueeze(-1)
        
        t_emb = self.type_emb(type_ids)
        sem_emb = self.semantic_emb(semantic_ids) # [Batch, Hidden]
        s_emb = self.shape_proj(shape_vecs)
        p_emb = self.param_cnt_proj(param_counts)
        
        concat = torch.cat([t_emb, sem_emb, s_emb, p_emb], dim=-1)
        return self.out_proj(concat)

class SetTransformerEncoder(nn.Module):
    def __init__(self, patch_dim, hidden_dim, num_inds=32, num_latents=32, num_heads=4, num_layers=1):
        super().__init__()
        
        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        layers = []
        for _ in range(num_layers):
            layers.append(ISAB(hidden_dim, hidden_dim, num_heads=num_heads, num_inds=num_inds, ln=True))
        self.enc = nn.Sequential(*layers)
        self.pool = PMA(hidden_dim, num_heads=num_heads, num_seeds=num_latents, ln=True)
        self.weight_out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # added metadata embedder for layer type
        self.meta_embedder = ArchitectureEmbedder(hidden_dim)
        
        self.fusion_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_patches, metadata=None):
        x = self.input_proj(x_patches)
        x = self.enc(x)
        latents = self.pool(x)
        w_emb = self.weight_out_proj(latents.mean(dim=1)) 
        
        if metadata is not None:
            if isinstance(metadata, list): pass 
            else:
                topo_emb = self.meta_embedder(
                    metadata['type_id'], 
                    metadata['semantic_id'], 
                    metadata['shape_vec'], 
                    metadata['param_count']
                )
                return self.fusion_norm(w_emb + topo_emb)
        
        return w_emb
"""
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
"""
