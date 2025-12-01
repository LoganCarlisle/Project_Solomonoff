#hyper net data handling
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoModelForCausalLM
from typing import List, Dict, Tuple, Any
from .ops import *

class HypernetDataset(Dataset):
    """
    A Dataset that represents a Neural Network as a Sequence of Layers with each layer getting patched.
    
    Flow:
    1. Load Model from hugging face be it gpt2 or whatever
    2. Patchify each layer -> Sequence of tokens [N_Patches, 4096].
    3. Serve the sequence + metadata(skeleton of the model) for the hypernet.
    """
    def __init__(self, model_id: str, patch_size: int = 64, device: str = "cpu"):
        self.model_id = model_id
        self.patch_size = patch_size
        self.device = device # Storage device using cpu to save vram
        
        print(f"[{self.model_id}] Initializing Hypernet Dataset")
        self.layers, self.metadata = self._load_sequence()

    def _load_sequence(self) -> Tuple[List[torch.Tensor], List[Dict]]:
        # Load Model Weights
        # Use CausalLM to ensure we capture the LM Head critical for generation
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
        except:
            print(f"  - Warning: CausalLM load failed, using base AutoModel.")
            model = AutoModel.from_pretrained(self.model_id)
            
        layers_list = []
        meta_list = []
        
        # 2. Process Layer-by-Layer
        # named_parameters() yields weights in topological order (wte -> blocks -> ln_f -> head)
        for idx, (name, param) in enumerate(model.named_parameters()):
            
            # Detach to remove grad history, move to CPU storage
            raw_tensor = param.detach().to(self.device)
            
            # Patchify: (H, W) -> (Num_Patches, Patch_Size^2)
            patches, orig_shape = patchify_tensor(raw_tensor, self.patch_size)
            
            layers_list.append(patches)
            
            meta_list.append({
                "layer_idx": idx,
                "name": name,
                "original_shape": orig_shape, # (H, W) - The "Canvas Size" for DiT
                "num_patches": patches.shape[0],
                "patch_dim": patches.shape[1]
            })
            
        print(f"[{self.model_id}] Loaded {len(layers_list)} layers.")
        print(f"  - Largest Layer: {max(m['num_patches'] for m in meta_list)} patches")
        print(f"  - Smallest Layer: {min(m['num_patches'] for m in meta_list)} patches")
        
        return layers_list, meta_list

    def __len__(self):
        # We treat 1 Model as 1 Sample. 
        # if multiple models this would be more maybe later
        return 1

    def __getitem__(self, idx):
        """
        Returns the full sequence for the model.
        The Training Loop handles the iteration Step t -> Step t+1 for each layer.
        """
        return {
            "model_id": self.model_id,
            "layers": self.layers,    # List[Tensor] - The Data
            "metadata": self.metadata # List[Dict]   - The Prompt/Bounds
        }
