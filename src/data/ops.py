#patchify logic and also the low rank upscaler
import torch
import torch.nn.functional as F
from typing import Tuple

def expand_low_rank(A: torch.Tensor, B: torch.Tensor, scaling_factor: float = 1.0) -> torch.Tensor:#gonna need to ad a complementary function and rank levels
    """
    Expands Low-Rank factors back to full weight matrix.
    Follows standard LoRA convention: W = B @ A * scale
    
    Args:
        A: The 'Down' projection [Rank, In_Dim]
        B: The 'Up' projection [Out_Dim, Rank]
        scaling_factor: Alpha / Rank (optional)
        
    Returns:
        W: [Out_Dim, In_Dim]
    """
    # Ensure they are 2D
    if A.dim() == 1: A = A.unsqueeze(0)
    if B.dim() == 1: B = B.unsqueeze(1)
    
    # Perform multiplication
    # B [Out, R] x A [R, In] -> [Out, In]
    W = (B @ A) * scaling_factor
    
    return W

def patchify_tensor(tensor: torch.Tensor, patch_size: int = 64) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Converts a 2D parameter tensor into a sequence of flat patches.
    This is the pre-processing step for the Set Transformer.
    
    Args:
        tensor: Shape (H, W) or (H,). 1D tensors (biases) are treated as (1, W).
        patch_size: The side length of the square patch (e.g., 64).
        
    Returns:
        patches: Shape (Num_Patches, Patch_Area). Patch_Area = 64*64 = 4096.
        original_shape: Tuple (H, W) needed for the DiT Fabricator to generate coords.
    """
    # 1. convert to 2d
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0) 
    
    H, W = tensor.shape
    
    #Calculate Padding calc padding--> pad matrix so its divisble byt patch size for our set former
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        # F.pad args are (Left, Right, Top, Bottom)
        tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)
    else:
        tensor_padded = tensor

    H_pad, W_pad = tensor_padded.shape
    
    # Patchify View -> Permute -> Flatten
    n_h = H_pad // patch_size
    n_w = W_pad // patch_size
    
    # Reshape to (Grid_H, Patch_H, Grid_W, Patch_W)
    reshaped = tensor_padded.view(n_h, patch_size, n_w, patch_size)
    
    # Permute to (Grid_H, Grid_W, Patch_H, Patch_W)
    transposed = reshaped.permute(0, 2, 1, 3)
    
    # Flatten to sequence: (Num_Patches, Patch_Area)
    patches = transposed.contiguous().view(-1, patch_size * patch_size)
    
    return patches, (H, W)

def unpatchify_tensor(patches: torch.Tensor, original_shape: Tuple[int, int], patch_size: int = 64) -> torch.Tensor:
    """
    Reconstructs the 2D tensor from patches. 
    Used during Inference/Generation to turn DiT output back into weights.
    """
    H, W = original_shape
    
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    H_pad, W_pad = H + pad_h, W + pad_w
    n_h, n_w = H_pad // patch_size, W_pad // patch_size
    
    # (N, P*P) -> (Grid_H, Grid_W, P, P)
    grid = patches.view(n_h, n_w, patch_size, patch_size)
    
    # (Grid_H, Grid_W, P, P) -> (Grid_H, P, Grid_W, P) -> (H_pad, W_pad)
    restored = grid.permute(0, 2, 1, 3).contiguous().view(H_pad, W_pad)
    
    return restored[:H, :W]
