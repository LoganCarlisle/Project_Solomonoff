import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
import math
  
class RealNLPWrapper(nn.Module):
    """Wraps a real Hugging Face model and its specific tokenizer."""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f" Failed to load '{model_name}' from Hugging Face. "
                               f"Check your internet connection or verify the model name. Error: {e}")

        self.model.config.return_dict = False
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs[0]

def load_real_nlp_zoo(device):
    """Downloads actual pretrained models to build our dataset."""
    print("\n" + "="*50)
    print(" DOWNLOADING AND ANALYZING HUGGING FACE ZOO")
    print("="*50)

    # Expanded the dataset with more diverse models
    model_names = [
        "prajjwal1/bert-tiny",               # 2 layers, 128 hidden (FFN 512)
        "google/bert_uncased_L-2_H-128_A-2", # 2 layers, 128 hidden
        "prajjwal1/bert-mini",               # 4 layers, 256 hidden (FFN 1024)
        "google/bert_uncased_L-4_H-128_A-2", # 4 layers, 128 hidden
        "google/bert_uncased_L-6_H-128_A-2", # 6 layers, 128 hidden
        "google/bert_uncased_L-2_H-256_A-4", # 2 layers, 256 hidden
        "google/bert_uncased_L-4_H-256_A-4", # 4 layers, 256 hidden
        "google/bert_uncased_L-6_H-256_A-4", # 6 layers, 256 hidden
    ]

    zoo = []
    total_params = 0

    for name in model_names:
        print(f"\n[+] Loading Architecture: {name}")
        wrapper = RealNLPWrapper(name).to(device)
        zoo.append(wrapper)

        model_params = sum(p.numel() for p in wrapper.parameters())
        total_params += model_params
        print(f"    Total Parameters: {model_params:,}")

    size_in_gb = (total_params * 4) / (1024 ** 3)
    print("\n" + "="*50)
    print(f" SUMMARY: {len(zoo)} Models Loaded")
    print(f" ZOO PARAMETERS: {total_params:,}")
    print(f" RAW DATASET SIZE: {size_in_gb:.4f} GB")
    print("="*50 + "\n")

    return zoo

def extract_weights_from_exported_model(hf_wrapper, max_layers, max_out, max_in):
    """
    Uses torch.export to get a static graph of the model, extracts the
    2D linear weight matrices, and pads them to fixed dimensions.
    """
    device = next(hf_wrapper.parameters()).device
    dummy_input = torch.randint(0, 2000, (1, 16)).to(device)
    exported_prog = export(hf_wrapper, (dummy_input,))

    weights_list = []
    state_dict = exported_prog.state_dict

    for name, param in state_dict.items():
        if len(param.shape) == 2:
            weights_list.append(param)

    padded_weights = []
    for w in weights_list:
        if len(padded_weights) >= max_layers:
            break

        out_dim, in_dim = w.shape
        out_dim_trunc = min(out_dim, max_out)
        in_dim_trunc = min(in_dim, max_in)

        padded = torch.zeros(max_out, max_in, device=w.device)
        padded[:out_dim_trunc, :in_dim_trunc] = w[:out_dim_trunc, :in_dim_trunc]
        padded_weights.append(padded)

    while len(padded_weights) < max_layers:
        padded_weights.append(torch.zeros(max_out, max_in, device=device))

    return torch.stack(padded_weights)

#git re-basin alignment of latent space length might switch to sinkhorn

def align_weights(reference_tensor, target_tensor):
    """
    Implements Weight Matching (Git Re-Basin).
    """
    aligned_tensor = target_tensor.clone()
    num_layers, max_out, max_in = target_tensor.shape

    for l in range(num_layers):
        ref_layer = reference_tensor[l].detach()
        tgt_layer = aligned_tensor[l].detach()

        if torch.all(ref_layer == 0) and torch.all(tgt_layer == 0):
            continue

        # 1. Compute Cost Matrix on GPU efficiently!
        cost_matrix = torch.cdist(ref_layer, tgt_layer, p=2.0)

        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

        aligned_layer = tgt_layer[col_ind]
        aligned_tensor[l] = aligned_layer

        if l + 1 < num_layers and not torch.all(aligned_tensor[l+1] == 0):
            next_layer = aligned_tensor[l+1].detach()
            aligned_next_layer = next_layer[:, col_ind]
            aligned_tensor[l+1] = aligned_next_layer

    return aligned_tensor
