#code from set transformer for position invariance
class MAB(nn.Module):
    
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.head_dim = dim_V // num_heads
        assert self.head_dim * num_heads == dim_V, "dim_V must be divisible by num_heads"

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()

        
        self.fc = nn.Sequential(
            nn.Linear(dim_V, dim_V * 4),
            nn.SiLU(),
            nn.Linear(dim_V * 4, dim_V)
        )

    def forward(self, Q, K):
        B, L_q, _ = Q.shape
        _, L_k, _ = K.shape

        # 1. Project and reshape into [Batch, Heads, Sequence, Head_Dim]
        q = self.fc_q(Q).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.fc_k(K).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.fc_v(K).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. FLASH ATTENTION (FLA)
        # PyTorch natively selects the most memory-efficient kernel (FlashAttention 2)
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # 3. Reshape back to [Batch, Sequence, Features]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.dim_V)
        O = self.fc_o(attn_output)

        # 4. Residual Connection & FFN
        Q_res = self.fc_q(Q)
        O = O + Q_res
        O = self.ln0(O)
        O = O + self.fc(O)
        O = self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        batch_size = X.size(0)
        S_batch = self.S.repeat(batch_size, 1, 1)
        return self.mab(S_batch, X)

#modules from wan
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.pad > 0:
            x = x[:, :, :-self.pad]
        return x

class ResidualCausalBlock(nn.Module):
    """Adds a ResNet-style skip connection and GroupNorm across the temporal dimension."""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = CausalConv1d(dim, dim, kernel_size=3)
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=3)
        self.norm2 = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        res = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + res)

class TemporalDownsample(nn.Module):
    """Wan-VAE's 2x Temporal Compression"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=2, stride=2)
    def forward(self, x):
        return self.conv(x)

class TemporalUpsample(nn.Module):
    """Reverses the 2x Temporal Compression using Learnable Deconvolution"""
    def __init__(self, dim):
        super().__init__()
        # UPGRADE: Replaced 'nearest' interpolation with a true learnable ConvTranspose!
        self.up_conv = nn.ConvTranspose1d(dim, dim, kernel_size=2, stride=2)
        self.conv = CausalConv1d(dim, dim, kernel_size=3)
    def forward(self, x):
        return self.conv(self.up_conv(x))
class WeightEncoder(nn.Module):
    def __init__(self, dim_in, dim_hidden, latent_dim, max_neurons_out, num_heads=8, num_seeds=64):
        super(WeightEncoder, self).__init__()
        self.num_seeds = num_seeds
        self.dim_hidden = dim_hidden

        self.proj_in_spatial = nn.Linear(dim_in, dim_hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, max_neurons_out, dim_hidden) * 0.02)

        self.sab1 = SAB(dim_hidden, dim_hidden, num_heads, ln=True)
        self.sab2 = SAB(dim_hidden, dim_hidden, num_heads, ln=True)
        self.pma = PMA(dim_hidden, num_heads, num_seeds, ln=True)

        self.res_block1 = ResidualCausalBlock(dim_hidden)
        self.downsample = TemporalDownsample(dim_hidden) # Wan-VAE 2x Down
        self.res_block2 = ResidualCausalBlock(dim_hidden)

        self.fc_mu = nn.Linear(dim_hidden, latent_dim)
        self.fc_logvar = nn.Linear(dim_hidden, latent_dim)

    def forward(self, X):
        B, T, N, D = X.shape

        # process each frame
        H_pooled_list = []
        for t in range(T):
            X_t = X[:, t, :, :] # Extract a single layer: [B, N, D]

            H_t = self.proj_in_spatial(X_t)
            H_t = H_t + self.pos_emb

            H_t = self.sab1(H_t)
            H_t = self.sab2(H_t)
            H_t_pooled = self.pma(H_t) # [B, num_seeds, dim_hidden]

            H_pooled_list.append(H_t_pooled)

        H_pooled = torch.stack(H_pooled_list, dim=1) # [B, T, num_seeds, dim_hidden]

        H_seq = H_pooled.view(B, T, self.num_seeds, self.dim_hidden)
        H_seq = H_seq.permute(0, 2, 3, 1).contiguous().view(B * self.num_seeds, self.dim_hidden, T)

        H_seq = self.res_block1(H_seq)
        H_seq = self.downsample(H_seq) # Compress temporal length (Layers) by half
        H_causal = self.res_block2(H_seq)

        H_causal = H_causal.view(B, self.num_seeds, self.dim_hidden, T // 2).permute(0, 3, 1, 2)

        mu = self.fc_mu(H_causal)
        logvar = self.fc_logvar(H_causal)
        return mu, logvar

class WeightDecoder(nn.Module):
    def __init__(self, latent_dim, dim_hidden, dim_out, num_target_neurons, num_heads=8, num_seeds=64):
        super(WeightDecoder, self).__init__()
        self.num_seeds = num_seeds
        self.dim_hidden = dim_hidden
        self.num_target_neurons = num_target_neurons

        self.proj_in = nn.Conv1d(latent_dim, dim_hidden, kernel_size=1)
        self.res_block1 = ResidualCausalBlock(dim_hidden)
        self.upsample = TemporalUpsample(dim_hidden) # Revert the 2x Down
        self.res_block2 = ResidualCausalBlock(dim_hidden)

        self.decoder_pos_emb = nn.Parameter(torch.randn(1, num_target_neurons, dim_hidden) * 0.02)

        self.cross_attn = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=True)
        self.sab = SAB(dim_hidden, dim_hidden, num_heads, ln=True)

        # expand by 4x
        self.fc_out = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden * 4),
            nn.SiLU(),
            nn.Linear(dim_hidden * 4, dim_out)
        )

    def forward(self, Z):
        B, T_half, S, L_dim = Z.shape
        T = T_half * 2

        Z_seq = Z.permute(0, 2, 3, 1).contiguous().view(B * S, L_dim, T_half)

        Z_seq = self.proj_in(Z_seq)
        Z_seq = self.res_block1(Z_seq)
        Z_seq = self.upsample(Z_seq) # Inflate temporal length back to full
        H_causal = self.res_block2(Z_seq)

        H_causal = H_causal.view(B, S, self.dim_hidden, T).permute(0, 3, 1, 2).contiguous()

        # decoding done here
        W_out_list = []
        for t in range(T):
            H_t_seeds = H_causal[:, t, :, :] # Extract single layer's temporal seeds: [B, S, dim_hidden]

            Q_batch = self.decoder_pos_emb.repeat(B, 1, 1) # [B, N, dim_hidden]

            H_set = self.cross_attn(Q_batch, H_t_seeds)
            H_set = self.sab(H_set)

            W_t_out = self.fc_out(H_set) 
            W_out_list.append(W_t_out)

        return torch.stack(W_out_list, dim=1)

class SpatioTemporalWeightVAE(nn.Module):
    def __init__(self, input_size, max_output_neurons, dim_hidden=256, latent_dim=64, num_heads=8, num_seeds=64): 
        super(SpatioTemporalWeightVAE, self).__init__()
        self.encoder = WeightEncoder(
            dim_in=input_size,
            dim_hidden=dim_hidden,
            latent_dim=latent_dim,
            max_neurons_out=max_output_neurons,
            num_heads=num_heads,
            num_seeds=num_seeds
        )
        self.decoder = WeightDecoder(
            latent_dim=latent_dim,
            dim_hidden=dim_hidden,
            dim_out=input_size,
            num_target_neurons=max_output_neurons,
            num_heads=num_heads,
            num_seeds=num_seeds
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X):
        mu, logvar = self.encoder(X)
        z = self.reparameterize(mu, logvar)
        X_recon = self.decoder(z)
        return X_recon, mu, logvar
