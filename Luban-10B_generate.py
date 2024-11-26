import torch
import argparse
import os
import time
import random
import numpy as np

from scripts.utils import activate_selective, load, load_jsonl

# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1)

# apply top-k and top-p (nucleus) filtering to logits
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Args:
        logits: Logits distribution with shape (batch size, vocabulary size).
        top_k >0: Keep only the top k tokens with the highest probability (top-k filtering).
        top_p >0.0: Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    batch_size, vocab_size = logits.size()

    if top_k > 0:
        top_k = min(top_k, vocab_size)  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        top_k_values, _ = torch.topk(logits, top_k)
        indices_to_remove = logits < top_k_values[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits = logits.masked_fill(sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove), filter_value)

    return logits

# generate text samples 
@torch.no_grad()
def sample_generate(model, tokenizer, input_ids, past_key_values, max_gen_len, temperature=1.0, top_k=50, top_p=0.95):
    eos_token_id = tokenizer.eos_token_id
    # bos_token_id = tokenizer.bos_token_id
    # pad_token_id = tokenizer.pad_token_id

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values

    logits = outputs.logits[:, -1, :]
    pred_token_idx = torch.multinomial(
        torch.nn.functional.softmax(logits / temperature, dim=-1), 
        num_samples=1
    )

    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :]
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        pred_token_idx = torch.multinomial(torch.nn.functional.softmax(filtered_logits, dim=-1), num_samples=1)

        if pred_token_idx.item() >= logits.size(-1):
            print(f"Error: pred_token_idx {pred_token_idx.item()} out of bounds for logits with size {logits.size(-1)}")
            break

        generated_ids.append(pred_token_idx.item())
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if len(generated_text) > pos:
            new_text = generated_text[pos:]
            print(new_text, end="", flush=True)
            pos = len(generated_text)

        if pred_token_idx.item() == eos_token_id:
            break
        elif pred_token_idx.item() == tokenizer.encode("\n")[0]:
            break
    print()
    return past_key_values

# perform selective inference
@torch.no_grad()
def selective_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=100):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = prompt
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = input_ids.to(model.device)
        
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        past_key_values = sample_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len,
            temperature=1.0, top_k=50, top_p=0.9
        )

def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    test_filepath = os.path.join(args.data_root, "machine.jsonl")
    print(f"Loading data from {test_filepath} ...")

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    adjusted_mask_size = args.max_gen_len + args.mask_size

    if args.activate_selective:
        kv_cache = activate_selective(
            model, mask_size=adjusted_mask_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    selective_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
        max_gen_len=args.max_gen_len  
    )

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="../Luban-10B-v2/checkpoint-500"
    )
    parser.add_argument("--data_root", type=str, default="test_data/")
    parser.add_argument("--activate_selective", action="store_true")
    parser.add_argument("--mask_size", type=int, default=512)
    parser.add_argument("--recent_size", type=int, default=512)
    parser.add_argument("--max_gen_len", type=int, default=100)
    args = parser.parse_args()

    main(args)
    end_time = time.time()
    modified_time = end_time - start_time
    print(f"Modified attention computation time: {modified_time} seconds")
