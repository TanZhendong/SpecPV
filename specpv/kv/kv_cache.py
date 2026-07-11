import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import StaticCache, OffloadedStaticCache
from dataclasses import dataclass
import math
from ..models.modeling_llama_kv import repeat_kv

class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


@dataclass
class CacheConfig:
    block_size: int = 16              
    n_sink_blocks: int = 2             
    n_retrieval_blocks: int = 256      
    n_window_blocks: int = 8       
    n_spec_tokens_buf: int = 128       
    max_batch_size: int = 1            
    selector: str = "quest"

    @property
    def sink_size(self) -> int:
        return self.n_sink_blocks * self.block_size

    @property
    def retrieval_size(self) -> int:
        return self.n_retrieval_blocks * self.block_size
    
    @property
    def window_size(self) -> int:
        return self.n_window_blocks * self.block_size

    @property
    def total_budget(self) -> int:
        return self.static_kv_size + self.n_spec_tokens_buf

    @property
    def static_kv_size(self) -> int:
        """Number of persistent tokens visible to partial attention."""
        return self.sink_size + self.retrieval_size + self.window_size


class PartialKVCache(Cache):
    def __init__(self, cache_config: CacheConfig, model_config, max_length, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        # config
        self.cache_config = cache_config
        self.model_config = model_config
        self.dtype = dtype
        self.num_layers = model_config.num_hidden_layers
        self.num_kv_heads = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)
        self.head_dim = getattr(model_config, "head_dim", model_config.hidden_size // model_config.num_attention_heads)
        # state flag
        self.retrieval_initialized = False
        self.enabled = False

        self.key_cache: list[dict[str, torch.Tensor]] = []
        self.value_cache: list[dict[str, torch.Tensor]] = []
        self.verified_lens = []
        self.global_verified_lens = 0
        max_blocks = math.ceil(max_length / cache_config.block_size)
        self.key_states_summary: list[dict[str, torch.Tensor]] = []
        self.summary_block_count = []

        # build per-layer cache
        for _ in range(self.num_layers):
            # allocate one big tensor [batch, heads, total_budget, head_dim]
            cache_shape = (cache_config.max_batch_size, self.num_kv_heads, cache_config.total_budget, self.head_dim)

            key_storage = torch.zeros(cache_shape, dtype=dtype, device=device)
            value_storage = torch.zeros(cache_shape, dtype=dtype, device=device)

            # offsets
            offset = 0
            sink_slice = slice(offset, offset + cache_config.sink_size); offset += cache_config.sink_size
            retri_slice = slice(offset, offset + cache_config.retrieval_size); offset += cache_config.retrieval_size
            window_slice = slice(offset, offset + cache_config.window_size); offset += cache_config.window_size
            buffer_slice = slice(offset, offset + cache_config.n_spec_tokens_buf)

            # record partitions
            self.key_cache.append({
                "all": key_storage,
                "sink": key_storage[:, :, sink_slice, :],
                "retrieval": key_storage[:, :, retri_slice, :],
                "window": key_storage[:, :, window_slice, :],
                "buffer": key_storage[:, :, buffer_slice, :]
            })
            self.value_cache.append({
                "all": value_storage,
                "sink": value_storage[:, :, sink_slice, :],
                "retrieval": value_storage[:, :, retri_slice, :],
                "window": value_storage[:, :, window_slice, :],
                "buffer": value_storage[:, :, buffer_slice, :]
            })
            self.key_states_summary.append({
                "max": torch.zeros((cache_shape[0], cache_shape[1], max_blocks, cache_shape[3]), device=device, dtype=dtype),
                "min": torch.zeros((cache_shape[0], cache_shape[1], max_blocks, cache_shape[3]), device=device, dtype=dtype),
            })
            self.summary_block_count.append(0)
            self.verified_lens.append(0)

    # from ..speculate.profile import record_time
    # @record_time("summary_key_states")
    def summary_key_states(self, key_states, seq_len, layer_idx: int):
        """
        Incrementally update the max/min values of summary_key_states[layer_idx].
        key_states: [B, H, L, D] represents the key of the current layer.
        """
        block_size = self.cache_config.block_size
        sink_size = self.cache_config.sink_size
        existing_blocks = self.summary_block_count[layer_idx]
        expected_blocks = max(0, (seq_len - sink_size) // block_size)

        if expected_blocks <= existing_blocks:
            return
        # update incrementally
        start = sink_size + existing_blocks * block_size
        end = start + (expected_blocks - existing_blocks) * block_size
        new_keys = key_states[:, :, start:end, :]              # [B,H,L_new,D]
        B, H, L, D = new_keys.shape
        n_blocks = L // block_size
        new_keys = new_keys[:, :, :n_blocks * block_size, :].reshape(B, H, n_blocks, block_size, D)
        k_max = new_keys.max(dim=3).values                     # [B,H,n_blocks,D]
        k_min = new_keys.min(dim=3).values
        dst_max = self.key_states_summary[layer_idx]["max"][:, :, existing_blocks:existing_blocks + n_blocks, :]
        dst_min = self.key_states_summary[layer_idx]["min"][:, :, existing_blocks:existing_blocks + n_blocks, :]
        dst_max.copy_(k_max)
        dst_min.copy_(k_min)
        self.summary_block_count[layer_idx] += n_blocks

    def add_to_sink(self, key_states, value_states, layer_idx: int):
        sink_size = self.cache_config.sink_size
        k = key_states[:, :, :sink_size, :]
        v = value_states[:, :, :sink_size, :]
        self.key_cache[layer_idx]["sink"][:, :, :sink_size, :] = k
        self.value_cache[layer_idx]["sink"][:, :, :sink_size, :] = v

    # from ..speculate.profile import record_time
    # @record_time("refresh_kv_cache")
    def refresh_retrieval(self, query_states, key_states, value_states, seq_len, layer_idx: int, reduce_type: str="mean"):
        # Sliding-window and StreamingLLM have no retrieved partition.  They
        # still use this refresh path to replace the recent window.
        if self.cache_config.retrieval_size == 0:
            window_size = self.cache_config.window_size
            assert seq_len >= window_size, (
                f"Window size {window_size} is larger than current sequence length {seq_len}."
            )
            win_slice = slice(seq_len - window_size, seq_len)
            self.key_cache[layer_idx]["window"].copy_(
                key_states[:, :, win_slice, :].to(self.key_cache[layer_idx]["window"].device, non_blocking=True)
            )
            self.value_cache[layer_idx]["window"].copy_(
                value_states[:, :, win_slice, :].to(self.value_cache[layer_idx]["window"].device, non_blocking=True)
            )
            self.verified_lens[layer_idx] = 0
            return

        # 1. update summary_key_states
        retrieval_len = seq_len - self.cache_config.window_size
        self.summary_key_states(key_states, retrieval_len, layer_idx)
        summary = self.key_states_summary[layer_idx]
        num_blocks = self.summary_block_count[layer_idx]

        # 2. Score each non-sink block. Quest uses the current queries and
        # per-dimension max/min summaries. H2O here is deliberately a
        # query-free baseline: it ranks blocks by the mean L2 norm of keys.
        if self.cache_config.selector == "quest":
            n_rep = int(query_states.size(1) // key_states.size(1))
            summary_max = repeat_kv(summary["max"][:, :, :num_blocks, :], n_rep)
            summary_min = repeat_kv(summary["min"][:, :, :num_blocks, :], n_rep)
            sim_max = torch.einsum("bhqd,bhnd->bhqn", query_states, summary_max)
            sim_min = torch.einsum("bhqd,bhnd->bhqn", query_states, summary_min)
            scores = torch.maximum(sim_max, sim_min)
            if reduce_type == "max":
                scores = scores.max(dim=2).values
            elif reduce_type == "mean":
                scores = scores.mean(dim=2)
            else:
                raise ValueError(f"Unknown reduce_type: {reduce_type}")
            if n_rep > 1:
                # Average query-head scores back to KV heads for GQA.
                B, H, N = scores.shape
                scores = scores.reshape(B, H // n_rep, n_rep, N).mean(dim=2)
        elif self.cache_config.selector == "h2o":
            block_size = self.cache_config.block_size
            candidate_keys = key_states[
                :, :, self.cache_config.sink_size : self.cache_config.sink_size + num_blocks * block_size, :
            ]
            candidate_keys = candidate_keys.reshape(
                candidate_keys.size(0), candidate_keys.size(1), num_blocks, block_size, candidate_keys.size(-1)
            )
            scores = torch.linalg.vector_norm(candidate_keys.float(), dim=-1).mean(dim=-1)
        else:
            raise ValueError(f"Unknown KV selector: {self.cache_config.selector}")

        # 4. top-k block selection
        topk_blocks = min(self.cache_config.n_retrieval_blocks, num_blocks)
        assert num_blocks >= topk_blocks, (
            f"Not enough blocks for retrieval: need {topk_blocks}, but only have {num_blocks}. "
            "Ensure seq_len and block partitioning produce enough blocks before refresh_retrieval."
        )
        _, top_indices = torch.topk(scores, k=topk_blocks, dim=-1)  # [B,H,K]
        # get shape
        block_size = self.cache_config.block_size
        B, H, K = top_indices.shape
        D = key_states.size(-1)
        # convert block indices → token indices
        token_offsets = torch.arange(block_size, device=top_indices.device)  # [block_size]
        token_indices = self.cache_config.sink_size + top_indices[..., None] * block_size + token_offsets  # [B,H,K,block_size]
        token_indices = token_indices.reshape(B, H, K * block_size)          # [B,H,K*block_size]
        token_indices, _ = torch.sort(token_indices, dim=-1)
        token_indices_exp = token_indices.unsqueeze(-1).expand(-1, -1, -1, D)  # [B,H,K*block_size,D] expand dim
        # indices keys/values
        retri_k_new = torch.take_along_dim(key_states, token_indices_exp, dim=2)  # [B,H,R,D]
        retri_v_new = torch.take_along_dim(value_states, token_indices_exp, dim=2)
        # write retrieval cache
        retri_k = self.key_cache[layer_idx]["retrieval"]
        retri_v = self.value_cache[layer_idx]["retrieval"]
        retri_k.copy_(retri_k_new.to(retri_k.device, non_blocking=True))
        retri_v.copy_(retri_v_new.to(retri_v.device, non_blocking=True))

        # 5. fill window with the most recent tokens
        win_k = self.key_cache[layer_idx]["window"]   # [B,H,W,D]
        win_v = self.value_cache[layer_idx]["window"]
        window_size = self.cache_config.window_size

        assert seq_len >= window_size, (
            f"Window size {window_size} is larger than current sequence length {seq_len}. "
            "Ensure seq_len >= window_size before calling refresh_retrieval."
        )
        win_slice = slice(seq_len - window_size, seq_len)
        win_k.copy_(key_states[:, :, win_slice, :].to(win_k.device, non_blocking=True))
        win_v.copy_(value_states[:, :, win_slice, :].to(win_v.device, non_blocking=True))

        # 6. reset verified_lens = 0
        self.verified_lens[layer_idx] = 0

    def init_key_values(self, full_past_key_values):
        if self.enabled:
            return
        for layer_idx, (key_states, value_states) in enumerate(full_past_key_values):
            self.add_to_sink(key_states, value_states, layer_idx)
        self.enabled = True

    def update(self, new_key_states, new_value_states, layer_idx: int):
        # buffer slice for this layer
        buf_k = self.key_cache[layer_idx]["buffer"]
        buf_v = self.value_cache[layer_idx]["buffer"]
        seq_len = new_key_states.size(2)
        start = self.verified_lens[layer_idx]
        end = start + seq_len  # new key: [B, H, L, D]
        assert end <= buf_k.size(2), (
            f"Buffer overflow at layer {layer_idx}: "
            f"trying to write {seq_len} tokens, but buffer size={buf_k.size(2)}, verified={start}."
        )
        # copy into buffer
        buf_k[:, :, start:end, :].copy_(new_key_states.to(buf_k.device, non_blocking=True))
        buf_v[:, :, start:end, :].copy_(new_value_states.to(buf_v.device, non_blocking=True))
        # return key and values
        valid_len = self.get_seq_length(layer_idx) + seq_len
        return (
            self.key_cache[layer_idx]["all"][:, :, :valid_len, :],
            self.value_cache[layer_idx]["all"][:, :, :valid_len, :]
        )

    def get_seq_length(self, layer_idx: int = 0):
        config = self.cache_config
        return config.sink_size + config.retrieval_size + config.window_size + self.verified_lens[layer_idx]

    def reset(self):
        self.retrieval_initialized = False
        self.enabled = False
        # reset per-layer caches
        for layer_idx in range(self.num_layers):
            # zero out key/value cache
            self.key_cache[layer_idx]["all"].zero_()
            self.value_cache[layer_idx]["all"].zero_()
            # reset summary stats
            self.key_states_summary[layer_idx]["max"].zero_()
            self.key_states_summary[layer_idx]["min"].zero_()
            self.summary_block_count[layer_idx] = 0
            self.verified_lens[layer_idx] = 0

    def reset_buffer(self):
        for layer_idx in range(self.num_layers):
            self.verified_lens[layer_idx] = 0
            self.key_cache[layer_idx]["buffer"].zero_()
            self.value_cache[layer_idx]["buffer"].zero_()


# from pytorch_memlab import profile
# @profile
# from ..speculate.profile import record_time
# @record_time("init_kv_cache")
def initialize_past_key_values(model, draft_model, cache_config, max_length=8192):
    config = model.config
    offloading = cache_config.enable_offload

    # init full kv cache
    full_past_key_values = StaticCache(
        config=config,
        max_cache_len=max_length,
        offloading=offloading
    )
    
    # init partial kv cache
    partial_cache_config = CacheConfig(
        block_size=cache_config.block_size, 
        n_sink_blocks=cache_config.n_sink_blocks,
        n_retrieval_blocks=cache_config.n_retrieval_blocks, 
        n_window_blocks=cache_config.n_window_blocks,
        n_spec_tokens_buf=cache_config.partial_spec_tokens + draft_model.total_tokens + 1,
        selector=cache_config.kv_selector,
    )
    partial_past_key_values = PartialKVCache(cache_config=partial_cache_config, model_config=config, dtype=model.dtype, max_length=max_length, device=next(model.parameters()).device)

    # init draft kv cache
    draft_past_key_values = StaticCache(
        config=draft_model.config,
        max_cache_len=max_length,
    )

    return full_past_key_values, partial_past_key_values, draft_past_key_values
