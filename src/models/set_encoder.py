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
class SetTransformerEncoder(nn.Module):
    """
    encode bariable patches using isab + pma pooling heavy work and tweaking needed for finding the right encoder architeture
    """
    def __init__(self, patch_dim, hidden_dim, num_inds=32, num_latents=64):
        super().__init__()
        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        
        # induced self attention block so its encoded invariant
        self.enc = nn.Sequential(
            ISAB(hidden_dim, hidden_dim, num_heads=4, num_inds=num_inds, ln=True),
            ISAB(hidden_dim, hidden_dim, num_heads=4, num_inds=num_inds, ln=True)
        )
        
        #  Pooling Step (PMA)
        # this summarizes the set into fixed vectors
        self.pool = PMA(hidden_dim, num_heads=4, num_seeds=num_latents, ln=True)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_patches):
        # Project
        x = self.input_proj(x_patches) # [B, N, Hidden]
        
        # Interact Self-Attention approximation
        x = self.enc(x)
        
        #  Pool Summarize
        latents = self.pool(x) # [B, Num_Latents, Hidden]
        
        # Collapse to single vector for Mamba or keep as sequence if Mamba supports it, here we mean, depending on the architure changes
        global_state = latents.mean(dim=1)
        
        return self.out_proj(global_state)
"""
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
"""
