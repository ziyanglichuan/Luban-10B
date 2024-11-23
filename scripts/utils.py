import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import json


def load(model_name_or_path):
    """Loads a pre-trained model and tokenizer from the specified path."""
    print(f"Loading model from {model_name_or_path} ...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
    )
    
    # Load model 
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval() 
    return model, tokenizer


def load_jsonl(file_path):
    """Loads data from a JSONL (JSON Lines) file."""
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line)) 
    return list_data_dict

def activate_selective(model, mask_size, recent_size):
    """Activates the SelectiveKQVCache for the given model."""
    k_seq_dim = v_seq_dim = 2  # Set dimensions for key and value sequences
    from scripts.model_architecture import (
        enable_llama_pos_shift_attention,  # Import function to enable position shift attention
    )

    enable_llama_pos_shift_attention(model)  # Apply position shift attention to the model

    # Initialize the SelectiveKQVCache with specified parameters
    kv_cache = SelectiveKQVCache(
        mask_size = mask_size,
        recent_size=recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    return kv_cache

class SelectiveKQVCache:
    """Selective Key-Value-Query Cache for managing past key-value pairs during model inference."""
    def __init__(self, mask_size=512, recent_size=512, k_seq_dim=2, v_seq_dim=2, q_seq_dim=2):
        print(f"SelectiveKQVCache: mask_size={mask_size}, recent_size={recent_size}")
        self.mask_size = mask_size
        self.recent_size = recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.q_seq_dim = q_seq_dim

    def _process_key_values(self, past_key_values, num_keep, return_truncated=False):
        """Processes and filters the past key-value pairs based on attention scores."""
        new_key_values = []
        for kv in past_key_values:
            k, v, q = kv
            
            device = k.device
            seq_len = k.size(self.k_seq_dim)
            recent_indices = torch.arange(seq_len - self.recent_size, seq_len, device=device)
            
            # Compute attention scores and truncate to remove the most recent part
            attn_scores = torch.matmul(q, k.transpose(-2, -1)).mean(dim=-1)
            truncated_avg_scores = attn_scores[:, :, :-self.recent_size]
            truncated_k = k[:, :, :-self.recent_size]
            truncated_v = v[:, :, :-self.recent_size]
            truncated_q = q[:, :, :-self.recent_size]

            # Select top key-value-query pairs based on attention scores
            if truncated_avg_scores.size(-1) > 0:
                top_indices = torch.topk(truncated_avg_scores, num_keep, dim=-1, largest=True).indices
                top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, k.size(-1))
                new_k_top = torch.gather(truncated_k, 2, top_indices_expanded)
                new_v_top = torch.gather(truncated_v, 2, top_indices_expanded)
                new_q_top = torch.gather(truncated_q, 2, top_indices_expanded)
            else:
                new_k_top = new_v_top = new_q_top = torch.empty(0, device=device, dtype=k.dtype)

            # Select the most recent key-value-query pairs
            new_k_recent = k[:, :, recent_indices]
            new_v_recent = v[:, :, recent_indices]
            new_q_recent = q[:, :, recent_indices]

            # Combine the top and recent key-value-query pairs
            new_k = torch.cat((new_k_top, new_k_recent), dim=2) if new_k_top.numel() > 0 else new_k_recent
            new_v = torch.cat((new_v_top, new_v_recent), dim=2) if new_v_top.numel() > 0 else new_v_recent
            new_q = torch.cat((new_q_top, new_q_recent), dim=2) if new_q_top.numel() > 0 else new_q_recent

            if return_truncated:
                new_key_values.append([truncated_k, truncated_v, truncated_q])
            else:
                new_key_values.append([new_k, new_v, new_q])

        return new_key_values

    def __call__(self, past_key_values):
        """Applies the selective caching process to the provided key-value pairs."""
        if past_key_values is None:
            return None

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.mask_size + self.recent_size:
            return past_key_values

        return self._process_key_values(past_key_values, self.mask_size)

    def evict_for_space(self, past_key_values, num_coming):
        """Evicts old key-value pairs to make space for new ones."""
        if past_key_values is None:
            return None

        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.mask_size + self.recent_size:
            return past_key_values

        num_keep = self.mask_size - num_coming
        return self._process_key_values(past_key_values, num_keep)
