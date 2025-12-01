# pytorch support for now is probaly not the move will denfintly need to fix this gonna use a huggingface config
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from typing import List, Dict, Union, Tuple
from src.data.ops import unpatchify_tensor, expand_low_rank

class ModelAssembler:
    """
    Utility to take generated weights/factors and rebuild a runnable Hugging Face model.
    """
    def __init__(self, model_id: str, device: str = "cpu"):
        self.config = AutoConfig.from_pretrained(model_id)
        self.device = device
        
    def create_skeleton(self):
        """Creates an initialized model with random weights."""
        print(f"Building Skeleton for {self.config._name_or_path}...")
        model = AutoModelForCausalLM.from_config(self.config)
        model.to(self.device)
        return model

    def assemble(self, 
                 generated_outputs: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]], 
                 metadata_list: List[Dict],
                 patch_size: int = 64):
        """
        Main reconstruction function.
        
        Args:
            generated_outputs: List of outputs from Hypernet. Can be:
                - List of patched tensors [N, 4096]
                - List of low-rank tuples (A, B)
            metadata_list: The metadata list from the Dataset (contains parameter names and shapes).
            patch_size: Size used during patching.
            
        Returns:
            Runnable PyTorch model with injected weights.
        """
        model = self.create_skeleton()
        state_dict = model.state_dict()
        
        print("Injecting generated parameters...")
        
        for i, output in enumerate(generated_outputs):
            meta = metadata_list[i]
            name = meta['name']
            target_shape = meta['original_shape']
            
            # --- CASE 1: Low Rank Factors (Tuple of A, B) ---
            if isinstance(output, (tuple, list)) and len(output) == 2:
                # Assuming output is (A, B) or (Down, Up)
                factor_A, factor_B = output[0], output[1]
                
                # Expand: W = B @ A
                full_weight = expand_low_rank(factor_A, factor_B)
                
            # --- CASE 2: Full Patched Layer ---
            elif isinstance(output, torch.Tensor):
                # Unpatchify: [N, P*P] -> [H, W]
                full_weight = unpatchify_tensor(output, target_shape, patch_size)
                
            else:
                raise ValueError(f"Unknown output type for layer {name}")

            # Verification & Reshaping
            if full_weight.shape != torch.Size(target_shape):
                # Handle 1D bias edge case (Unpatchify returns 2D [1, W])
                if len(target_shape) == 1 and full_weight.shape == (1, target_shape[0]):
                    full_weight = full_weight.squeeze(0)
                else:
                    # Try to reshape if element count matches
                    if full_weight.numel() == torch.Size(target_shape).numel():
                         full_weight = full_weight.view(target_shape)
                    else:
                        print(f"Warning: Shape Mismatch for {name}. Target {target_shape}, Got {full_weight.shape}")

            # Injection
            if name in state_dict:
                state_dict[name].copy_(full_weight)
            else:
                print(f"Warning: {name} found in metadata but not in model state_dict.")

        return model

def verify_assembly(model, input_text="Hello world"):
    """Runs a quick forward pass to ensure the model doesn't crash."""
    from transformers import AutoTokenizer
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        print(f"Verification Pass Successful. Logits shape: {logits.shape}")
        return logits
    except Exception as e:
        print(f"Verification Failed: {e}")
        return None
