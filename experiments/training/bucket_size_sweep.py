import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig, Cache, FlashAttentionKwargs, apply_rotary_pos_emb, repeat_kv
from typing import Tuple, Optional, Unpack
from tqdm import tqdm
from source.scfa_paper.scfa_wrapper import hash_sparse_attention
import datasets
from copy import deepcopy
import os

def vector_hash_fn(x: torch.Tensor, num_buckets: int, R: torch.Tensor) -> torch.Tensor:
    """
    x: (..., D)
    R: (D, b/2)
    """
    D = x.shape[-1]
    assert R.shape == (D, num_buckets // 2)
    return torch.argmax(torch.cat([x @ R, -x @ R], dim=-1), dim=-1)

def get_vector_hash(D: int, num_buckets: int, device: torch.device = "cpu", dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    R = torch.randn(D, num_buckets // 2, device = device, dtype = dtype)
    return lambda x: vector_hash_fn(x, num_buckets, R)

def scfa_hash_attn(q, k, v, num_buckets: int, sm_scale: float = 1.0, vector_hash = None):
    """
    q: (B, H, N, D)
    k: (B, H, N, D)
    v: (B, H, N, D)
    """
    B, H, N, D = q.shape
    if vector_hash is None:
        vector_hash = get_vector_hash(D = D, num_buckets = num_buckets, device = q.device)
    assert num_buckets % 2 == 0, "num_buckets must be even"
    q_hashes = vector_hash(q).to(torch.int32).transpose(1, 2).contiguous() # (B, N, H)
    k_hashes = vector_hash(k).to(torch.int32).transpose(1, 2).contiguous() # (B, N, H)
    assert q_hashes.max() < num_buckets, "q_hashes must be less than num_buckets"
    assert k_hashes.max() < num_buckets, "k_hashes must be less than num_buckets"
    assert q_hashes.min() >= 0, "q_hashes must be non-negative"
    assert k_hashes.min() >= 0, "k_hashes must be non-negative"
    scfa_q = q.transpose(1, 2).contiguous() # (B, N, H, D)
    scfa_k = k.transpose(1, 2).contiguous() # (B, N, H, D)
    scfa_v = v.transpose(1, 2).contiguous() # (B, N, H, D)
    return hash_sparse_attention(scfa_q, scfa_k, scfa_v, q_hashes, k_hashes, sm_scale).transpose(1, 2).contiguous()

class HashAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, num_buckets: int, device):
        super().__init__(config, layer_idx)
        self.vector_hash = get_vector_hash(D = self.head_dim, num_buckets = num_buckets, device = device)
        self.num_buckets = num_buckets

    def forward(
        self,
        hidden_states: torch.Tensor, # (batch_size, seq_len, hidden_size)
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        """
        Shapes:
        query_states: (batch_size, num_heads, seq_len, head_dim)
        key_states: (batch_size, num_heads, seq_len, head_dim)
        value_states: (batch_size, num_heads, seq_len, head_dim)
        """

        try:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attn_output = scfa_hash_attn(
                q = query_states,
                k = repeat_kv(key_states, self.num_key_value_groups),
                v = repeat_kv(value_states, self.num_key_value_groups),
                num_buckets = self.num_buckets,
                sm_scale = self.scaling,
                vector_hash = self.vector_hash,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
        
        except Exception as e:
            print(query_states.shape, self.num_buckets)
            print(e)
        
        return attn_output, None
    
class LearnableHash(nn.Module):
    def __init__(self, D: int, num_buckets: int, device: torch.device = "cpu", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.D = D
        self.num_buckets = num_buckets
        self.device = device
        self.dtype = dtype
        self.R = nn.Parameter(torch.randn(D, num_buckets // 2, device = device, dtype = dtype), requires_grad = True)

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        proj = torch.matmul(x, self.R)
        logits = torch.cat([proj, -proj], dim=-1)
        one_hot = F.gumbel_softmax(logits, tau = tau, hard = True)
        return torch.argmax(one_hot, dim=-1)
    
class LearnableHashAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, num_buckets: int, device):
        super().__init__(config, layer_idx)
        self.vector_hash = LearnableHash(D = self.head_dim, num_buckets = num_buckets, device = device)
        self.num_buckets = num_buckets

    def forward(
        self,
        hidden_states: torch.Tensor, # (batch_size, seq_len, hidden_size)
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        """
        Shapes:
        query_states: (batch_size, num_heads, seq_len, head_dim)
        key_states: (batch_size, num_heads, seq_len, head_dim)
        value_states: (batch_size, num_heads, seq_len, head_dim)
        """

        try:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attn_output = scfa_hash_attn(
                q = query_states,
                k = repeat_kv(key_states, self.num_key_value_groups),
                v = repeat_kv(value_states, self.num_key_value_groups),
                num_buckets = self.num_buckets,
                sm_scale = self.scaling,
                vector_hash = self.vector_hash,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
        
        except Exception as e:
            print(query_states.shape, self.num_buckets)
            print(e)
        
        return attn_output, None
    
# Monkeypatch time
def monkeypatch(model, num_buckets: int, learnable_hash: bool = False, one_at_a_time: bool = False):
    n_modules_to_replace = len(list(filter(lambda x: isinstance(x, LlamaAttention), model.modules())))
    pbar = tqdm(total = n_modules_to_replace, desc = "Replacing attention modules")
    if one_at_a_time:
        pbar.disable = True    
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            # Construct new module
            if learnable_hash:
                new_attn_module = LearnableHashAttention(config = module.config, layer_idx = module.layer_idx, num_buckets = num_buckets, device = model.device)
            else:
                new_attn_module = HashAttention(config = module.config, layer_idx = module.layer_idx, num_buckets = num_buckets, device = model.device)
            new_attn_module.load_state_dict(module.state_dict(), strict = False)
            new_attn_module.to(model.device).to(torch.bfloat16)

            # Split full name to find parent module
            parent_module = model
            parent_name_parts = name.split('.')
            child_name = parent_name_parts[-1]

            if len(parent_name_parts) > 1:
                for part in parent_name_parts[:-1]:
                    if part.isdigit(): # Handles modules in nn.Sequential or nn.ModuleList
                        parent_module = parent_module[int(part)]
                    else:
                        parent_module = getattr(parent_module, part)

            setattr(parent_module, child_name, new_attn_module)
            if one_at_a_time:
                return
            pbar.update(1)

def train_model(model, tokenizer, num_buckets: int, learnable_hash: bool = False, num_steps_per_module: int = 1000, num_all_param_tune_steps: int = 10000, max_length: int = 512, ema_alpha: float = 0.05):
    fineweb_dataset = datasets.load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    hash_model = deepcopy(model).to(model.device)
    tokenizer.pad_token = tokenizer.eos_token

    total_steps_processed = 0
    n_modules_to_replace = len(list(filter(lambda x: isinstance(x, LlamaAttention), hash_model.modules())))
    while n_modules_to_replace > 0:
        print(f"{n_modules_to_replace} LlamaAttention modules remaining")
        # Patch llama attention modules into hash attention modules, one at a time
        monkeypatch(hash_model, num_buckets, learnable_hash = learnable_hash, one_at_a_time = True)
        n_modules_to_replace -= 1
        hash_model.train()

        # Freeze all but the hash attention modules
        for p in hash_model.parameters():
            p.requires_grad = False
        for name, module in hash_model.named_modules():
            if isinstance(module, HashAttention):
                for param in module.parameters():
                    param.requires_grad = True

        optimizer = torch.optim.AdamW(hash_model.parameters(), lr=1e-5)
        hash_model.train()

        step = 0
        loss_history = []
        perplexity_history = []
        ema_loss = None
        ema_perplexity = None
        # Use streaming dataset with tqdm
        with tqdm(total=num_steps_per_module, desc="Training") as pbar:
            for example in fineweb_dataset:
                if step >= num_steps_per_module:
                    total_steps_processed += step
                    fineweb_dataset = fineweb_dataset.skip(total_steps_processed) # continue from where we left off
                    break
                    
                text = example['text']
                inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                                truncation=True, padding=True)
                inputs = {k: v.to(hash_model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = hash_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update EMA metrics
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                    ema_perplexity = perplexity
                else:
                    ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss
                    ema_perplexity = ema_alpha * perplexity + (1 - ema_alpha) * ema_perplexity
                
                # Append current loss and perplexity to history
                loss_history.append(current_loss)
                perplexity_history.append(perplexity)
                
                step += 1
                
                # Update pbar
                pbar.set_description(f"Training - Loss: {ema_loss:.4f}, PPL: {ema_perplexity:.2f}")
                pbar.update(1)

    # Save model
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", f"[hash_tune_only]hash_model_buckets_{num_buckets}_learnable_hash_{learnable_hash}_steps_{int(num_steps_per_module)}_maxlength_{int(max_length)}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    hash_model.save_pretrained(checkpoint_dir)
    metrics = compute_fineweb_metrics(model, hash_model, tokenizer, max_length=max_length, num_samples=1000)
    torch.save(metrics, os.path.join(checkpoint_dir, "metrics.pt"))
    torch.save(loss_history, os.path.join(checkpoint_dir, "loss_history.pt"))
    torch.save(perplexity_history, os.path.join(checkpoint_dir, "perplexity_history.pt"))

    # Finetune full model (no frozen parameters)
    for p in hash_model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(hash_model.parameters(), lr=1e-5)
    hash_model.train()
    step = 0
    ema_loss = None
    ema_perplexity = None
    with tqdm(total=num_all_param_tune_steps, desc="Training") as pbar:
        for example in fineweb_dataset:
            if step >= num_all_param_tune_steps:
                total_steps_processed += step
                fineweb_dataset = fineweb_dataset.skip(total_steps_processed) # continue from where we left off
                break
                    
            text = example['text']
            inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                            truncation=True, padding=True)
            inputs = {k: v.to(hash_model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = hash_model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA metrics
            current_loss = loss.item()
            if ema_loss is None:
                ema_loss = current_loss
                ema_perplexity = perplexity
            else:
                ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss
                ema_perplexity = ema_alpha * perplexity + (1 - ema_alpha) * ema_perplexity
            
            # Append current loss and perplexity to history
            loss_history.append(current_loss)
            perplexity_history.append(perplexity)
            
            step += 1
            
            # Update pbar
            pbar.set_description(f"Training - Loss: {ema_loss:.4f}, PPL: {ema_perplexity:.2f}")
            pbar.update(1)

    # Save model
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", f"[allparam_tune]hash_model_buckets_{num_buckets}_learnable_hash_{learnable_hash}_steps_{int(num_steps_per_module)}_post_tune_steps_{int(num_all_param_tune_steps)}_maxlength_{int(max_length)}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics = compute_fineweb_metrics(model, hash_model, tokenizer, max_length=max_length, num_samples=1000)
    hash_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(metrics, os.path.join(checkpoint_dir, "metrics.pt"))
    torch.save(loss_history, os.path.join(checkpoint_dir, "loss_history.pt"))
    torch.save(perplexity_history, os.path.join(checkpoint_dir, "perplexity_history.pt"))

# Function to compute KL divergence between original and hash model
def compute_fineweb_metrics(original_model, hash_model, tokenizer, max_length=512, num_samples: int = 1000):
    # Load a slice of the FineWeb dataset
    fineweb_dataset = datasets.load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    fineweb_sample = fineweb_dataset.take(num_samples) 
    texts = [sample['text'] for sample in fineweb_sample]

    tokenizer.pad_token = tokenizer.eos_token
    kl_divergences = []
    original_perplexities = []
    hash_perplexities = []
    
    for text in tqdm(texts):
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        inputs = {k: v.to(original_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get logits from both models
            original_outputs = original_model(**inputs)
            hash_outputs = hash_model(**inputs)
            
            original_logits = original_outputs.logits
            hash_logits = hash_outputs.logits
            
            # Convert to probabilities
            original_probs = F.softmax(original_logits, dim=-1)
            hash_probs = F.softmax(hash_logits, dim=-1)
            
            # Compute KL divergence: KL(original || hash)
            kl_div = F.kl_div(hash_probs.log(), original_probs, reduction='batchmean')
            kl_divergences.append(kl_div.item())
            
            # Compute perplexities
            # Shift logits and labels for next token prediction
            shift_logits_original = original_logits[..., :-1, :].contiguous()
            shift_logits_hash = hash_logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            # Compute cross entropy loss
            original_loss = F.cross_entropy(shift_logits_original.view(-1, shift_logits_original.size(-1)), 
                                          shift_labels.view(-1), reduction='mean')
            hash_loss = F.cross_entropy(shift_logits_hash.view(-1, shift_logits_hash.size(-1)), 
                                      shift_labels.view(-1), reduction='mean')
            
            # Convert loss to perplexity
            original_perplexity = torch.exp(original_loss).item()
            hash_perplexity = torch.exp(hash_loss).item()
            
            original_perplexities.append(original_perplexity)
            hash_perplexities.append(hash_perplexity)
    
    return {
        'kl_divergences': kl_divergences,
        'original_perplexities': original_perplexities,
        'hash_perplexities': hash_perplexities
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--learnable_hash", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                torch_dtype=torch.bfloat16,
                                                device_map=args.device)
    buckets_sweep = [4, 8, 16, 32, 64]
    for num_buckets in buckets_sweep:
        train_model(model, 
                    tokenizer, 
                    num_buckets=num_buckets, 
                    num_steps_per_module=1e3, 
                    num_all_param_tune_steps=1e4, 
                    max_length=512, 
                    learnable_hash=args.learnable_hash)