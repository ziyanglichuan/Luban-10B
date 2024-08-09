import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

# Export only the "enable_llama_pos_shift_attention" function from this module
__all__ = ["enable_llama_pos_shift_attention"]

# apply rotary positional embeddings to a single input tensor
def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # Squeeze unnecessary dimensions from cos and sin, and apply positional embeddings
    cos, sin = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1), sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)

# Modified forward function for Llama's attention mechanism with position shift
def llama_pos_shift_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Determine the batch size and query length
    bsz, q_len, _ = hidden_states.size()
    
    # Helper function to project, split, and concatenate tensors for multi-head attention
    def project_split_and_cat(proj, tp_split):
        slices = proj.weight.split(tp_split, dim=0)
        return torch.cat([F.linear(hidden_states, slice) for slice in slices], dim=-1)

    # If pretraining tensor parallelism is enabled, split and project tensors accordingly
    if self.config.pretraining_tp > 1:
        tp_split = (self.num_heads * self.head_dim) // self.config.pretraining_tp
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        
        query_states = project_split_and_cat(self.q_proj, tp_split)
        key_states = project_split_and_cat(self.k_proj, key_value_slicing)
        value_states = project_split_and_cat(self.v_proj, key_value_slicing)
    else:
        query_states, key_states, value_states = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

    # Reshape and transpose the query, key, and value tensors
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Calculate the length of the key-value sequence
    kv_seq_len = key_states.shape[-2] + (past_key_value[0].shape[-2] if past_key_value is not None else 0)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    # Concatenate past key and value states if available
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
        query_states_temp = torch.cat([past_key_value[2], query_states], dim=2)
    else:
        query_states_temp = query_states

    # Cache the current key, value, and query states if use_cache is enabled
    past_key_value = (key_states, value_states, query_states_temp) if use_cache else None

    # Apply rotary positional embeddings to query and key states
    query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
    key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
    key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

    # Repeat key and value tensors for grouped multi-head attention
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Calculate attention weights using scaled dot-product
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # Ensure attention weights have the expected size
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")

    # Apply attention mask if provided
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
        attn_weights += attention_mask

    # Normalize attention weights and compute attention output
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Ensure attention output has the expected size
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"attn_output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}")

    # Reshape and project attention output
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)

    # Apply output projection, considering tensor parallelism if enabled
    if self.config.pretraining_tp > 1:
        tp_split = self.hidden_size // self.config.pretraining_tp
        attn_output = sum(F.linear(attn_output[i], self.o_proj.weight.split(tp_split, dim=1)[i]) for i in range(self.config.pretraining_tp))
    else:
        attn_output = self.o_proj(attn_output)

    # Return the attention output, and optionally the attention weights and past key-values
    return attn_output, attn_weights if output_attentions else None, past_key_value

# enable positional shift in Llama's attention across all attention modules in the model
def enable_llama_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, LlamaAttention):
            module.forward = types.MethodType(llama_pos_shift_attention_forward, module)
        elif len(list(module.children())) > 0:
            enable_llama_pos_shift_attention(module)
