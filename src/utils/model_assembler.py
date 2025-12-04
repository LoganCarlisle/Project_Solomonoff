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
        # Suppress generic warnings for cleaner logs
        import transformers
        transformers.logging.set_verbosity_error()
        model = AutoModelForCausalLM.from_config(self.config)
        model.to(self.device)
        return model
    """ Main reconstruction function. Args: generated_outputs: List of outputs from Hypernet. Can be: - List of patched tensors [N, 4096] 
    - List of low-rank tuples (A, B) metadata_list: 
    The metadata list from the Dataset (contains parameter names and shapes). 
    patch_size: Size used during patching. 
    Returns: Runnable PyTorch model with injected weights. """
    def assemble(self, 
                 generated_outputs: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]], 
                 metadata_list: List[Dict],
                 patch_size: int = 64):
        """
        Args:
            generated_outputs: List of patched tensors [N, 4096] or factors.
            metadata_list: The metadata list from the Dataset.
        """
        model = self.create_skeleton()
        state_dict = model.state_dict()
        
        # Iterate and inject
        for i, output in enumerate(generated_outputs):
            meta = metadata_list[i]
            name = meta['name']
            
            # Get Target Shape from METADATA Used for Unpatching
            # Note had issues creased out: This might be (1, 768) for a bias, even if model wants (768,)
            reconstruction_shape = meta['original_shape']
            
            # Reconstruct Unpatch  Expand
            if isinstance(output, (tuple, list)) and len(output) == 2:
                # Low Rank Case
                full_weight = expand_low_rank(output[0], output[1])
            else:
                # Direct Patch Case
                full_weight = unpatchify_tensor(output, reconstruction_shape, patch_size)
    
            # Injection Logic for weights
            if name in state_dict:
                param = state_dict[name]
                
               
                if full_weight.shape != param.shape:
                    if full_weight.numel() == param.numel():
                        full_weight = full_weight.view_as(param)
                    else:
                        print(f" Shape Mismatch for {name}: Gen {full_weight.shape} vs Param {param.shape}")
                        continue 
                
                # Copy values
                param.copy_(full_weight)
                
            else:
                pass 
    
        return model

def verify_assembly_simple(self, model, input_text="Hello world"):
        """runna quick logits check before running full perplexity."""
        try:
            tokenizer = self.text_validator.tokenizer
            inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            if torch.isnan(outputs.logits).any():
                print(" Logits contain NaNs")
                return False
            return True
        except Exception as e:
            print(f"Simple Verification Failed dam: {e}")
            return False
