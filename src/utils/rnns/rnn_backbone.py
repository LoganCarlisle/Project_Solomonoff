import torch
import torch.nn as nn
import torch.nn.functional as F

# using kernals from fla these are the imports that get wrapped for to keep previous state
#shoutout the fla peeps
# add citation eventually
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    HAS_DELTA = True
except ImportError:
    HAS_DELTA = False
    print("FLAWrapper ERROR: DeltaNet kernels not found.")

try:
    from fla.ops.gla import chunk_gla, fused_recurrent_gla
    HAS_GLA = True
except ImportError:
    HAS_GLA = False
    print("FLAWrapper ERROR: GLA kernels not found.")


class RobustGLA(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward_train(self, x, initial_state=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, -1)
        k = self.k_proj(x).view(B, L, self.num_heads, -1)
        v = self.v_proj(x).view(B, L, self.num_heads, -1)
        g = F.logsigmoid(self.g_proj(x)).view(B, L, self.num_heads, -1)
        
        if not HAS_GLA: raise ImportError("GLA Kernels missing")
        try:
            y, final_state = chunk_gla(q, k, v, g, initial_state=initial_state, output_final_state=True)
        except TypeError:
            y, final_state = fused_recurrent_gla(q, k, v, g, initial_state=initial_state, output_final_state=True)
        
        y = y.reshape(B, L, D)
        return self.o_proj(self.norm(y)), final_state

    def forward_step(self, x, prev_state=None):
        x_seq = x.unsqueeze(1)
        B, L, D = x_seq.shape
        q = self.q_proj(x_seq).view(B, L, self.num_heads, -1)
        k = self.k_proj(x_seq).view(B, L, self.num_heads, -1)
        v = self.v_proj(x_seq).view(B, L, self.num_heads, -1)
        g = F.logsigmoid(self.g_proj(x_seq)).view(B, L, self.num_heads, -1)
        
        if not HAS_GLA: raise ImportError("GLA Kernels missing")
        y, new_state = fused_recurrent_gla(q, k, v, g, initial_state=prev_state, output_final_state=True)
        
        return self.o_proj(self.norm(y.reshape(B, D))), new_state

class RobustDeltaNet(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward_train(self, x, initial_state=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, -1)
        k = self.k_proj(x).view(B, L, self.num_heads, -1)
        v = self.v_proj(x).view(B, L, self.num_heads, -1)
        beta = self.beta_proj(x).view(B, L, self.num_heads, -1)
        
        if not HAS_DELTA: raise ImportError("DeltaNet Kernels missing")
        try:
            y, final_state = chunk_gated_delta_rule(q, k, v, beta, initial_state=initial_state, output_final_state=True)
        except TypeError:
            y, final_state = fused_recurrent_gated_delta_rule(q, k, v, beta, initial_state=initial_state, output_final_state=True)
            
        y = y.reshape(B, L, D)
        return self.o_proj(self.norm(y)), final_state

    def forward_step(self, x, prev_state=None):
        x_seq = x.unsqueeze(1)
        B, L, D = x_seq.shape
        q = self.q_proj(x_seq).view(B, L, self.num_heads, -1)
        k = self.k_proj(x_seq).view(B, L, self.num_heads, -1)
        v = self.v_proj(x_seq).view(B, L, self.num_heads, -1)
        beta = self.beta_proj(x_seq).view(B, L, self.num_heads, -1)
        
        if not HAS_DELTA: raise ImportError("DeltaNet Kernels missing")
        y, new_state = fused_recurrent_gated_delta_rule(q, k, v, beta, initial_state=prev_state, output_final_state=True)
        return self.o_proj(self.norm(y.reshape(B, D))), new_state


class StackedRobustBackbone(nn.Module):
    """
    Stacks multiple robust layers with residual connections.
    Manages a list of states [S_0, S_1, ... S_N].
    """
    def __init__(self, layer_type, d_model, num_layers=4, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Instantiate layers based on type string use gdn for gated delta net
        for _ in range(num_layers):
            if layer_type == 'gla':
                self.layers.append(RobustGLA(d_model, num_heads=num_heads))
            elif layer_type in ['deltanet', 'gdn']:
                self.layers.append(RobustDeltaNet(d_model, num_heads=num_heads))
    
    def forward_train(self, x, initial_states=None):
        if initial_states is None:
            initial_states = [None] * len(self.layers)
            
        final_states = []
        current_x = x
        
        for i, layer in enumerate(self.layers):
            # run layer
            out_x, state = layer.forward_train(current_x, initial_state=initial_states[i])
            
            # residual Connection
            current_x = current_x + out_x
            final_states.append(state)
            
        return current_x, final_states

    def forward_step(self, x, prev_states=None):
        if prev_states is None:
            prev_states = [None] * len(self.layers)
            
        new_states = []
        current_x = x
        
        for i, layer in enumerate(self.layers):
            # Run layer
            out_x, state = layer.forward_step(current_x, prev_state=prev_states[i])
            
            # Residual Connection for seq gen
            current_x = current_x + out_x
            new_states.append(state)
            
        return current_x, new_states
