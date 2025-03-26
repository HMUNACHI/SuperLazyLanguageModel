import gc
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cactus.common import DTYPE
from cactus.config import CactusConfig
from cactus.nn.autodiff import (CactusBundledMatmulFunction,
                                CactusLoraFunction,
                                CactusLoraQKVLinearFunction,
                                CactusMatmulFunction)
from cactus.utils import load_tensor_from_storage


class CactusEmbedding(torch.nn.Module):
    def __init__(self, weight_path, vocab_size, hidden_size, padding_idx=None):
        super().__init__()
        self.weight_path = weight_path
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        with torch.no_grad():
            weight = load_tensor_from_storage(
                weight_path=self.weight_path,
                shape=(self.vocab_size, self.hidden_size),
                to_ram=False,
            )
        return weight[input_ids]


class CactusRotaryEmbedding(nn.Module):
    def __init__(self, config: CactusConfig):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        inv_freq, self.attention_scaling = self.compute_rope_parameters(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def compute_rope_parameters(
        self,
        config: Optional[CactusConfig] = None,
    ) -> Tuple["torch.Tensor", float]:
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        return inv_freq, attention_factor


class CactusRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, weight_path, eps=1e-6):
        super().__init__()
        weight = load_tensor_from_storage(
            weight_path=weight_path, shape=hidden_size, to_ram=True
        )
        self.register_buffer("weight", weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class CactusLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_path, bias_path=None):
        """
        CactusLinear lazily loads a large weight matrix from disk on every forward pass
        with gradients disabled. The bias, being much smaller, is loaded entirely into RAM.
        Both weight and bias are not tracked by autograd.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_path = weight_path
        self.bias_path = bias_path

        if self.bias_path is not None:
            with torch.no_grad():
                bias = load_tensor_from_storage(
                    weight_path=self.bias_path, shape=(self.out_features,), to_ram=True
                )
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        weight = load_tensor_from_storage(
            weight_path=self.weight_path,
            shape=(self.out_features, self.in_features),
            to_ram=False,
        ).t()

        Wx = CactusMatmulFunction.apply(x, weight, 1.0)
        if self.bias is not None:
            Wx += self.bias
        return Wx


class CactusMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = nn.SiLU()
        self.gate_proj_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.mlp.gate_proj.weight.bin"
        )
        self.up_proj_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.mlp.up_proj.weight.bin"
        )
        self.down_proj_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.mlp.down_proj.weight.bin"
        )

    def forward(self, x):
        # Removed no_grad wrappers.
        gate_proj = load_tensor_from_storage(
            weight_path=self.gate_proj_path,
            shape=(self.intermediate_size, self.hidden_size),
            to_ram=False,
        )
        up_proj = load_tensor_from_storage(
            weight_path=self.up_proj_path,
            shape=(self.intermediate_size, self.hidden_size),
            to_ram=False,
        )
        down_proj = load_tensor_from_storage(
            weight_path=self.down_proj_path,
            shape=(self.hidden_size, self.intermediate_size),
            to_ram=False,
        )

        bundles = [
            (x, gate_proj.t(), 1.0),
            (x, up_proj.t(), 1.0),
        ]
        gate_proj_out, up_proj_out = CactusBundledMatmulFunction.apply(bundles)
        activated_gate_proj = self.act_fn(gate_proj_out) * up_proj_out
        return CactusMatmulFunction.apply(activated_gate_proj, down_proj.t(), 1.0)


class CactusLoraLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        r,
        alpha,
        weight_path,
        bias_path=None,
        lora_dropout=0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r
        self.weight_path = weight_path
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.bias = None
        if bias_path is not None:
            with torch.no_grad():
                bias = load_tensor_from_storage(
                    weight_path=bias_path, shape=(self.out_features,), to_ram=True
                )
            self.register_buffer("bias", bias)

    def forward(self, x):
        x_dropped = self.lora_dropout(x)
        return CactusLoraFunction.apply(
            x_dropped,
            self.lora_A,
            self.lora_B,
            self.weight_path,
            self.scaling,
            self.bias,
        )


class CactusLoraQKVLinear(nn.Module):
    def __init__(
        self,
        config,
        head_dim,
        q_weight_path,
        k_weight_path,
        v_weight_path,
        q_bias_path=None,
        k_bias_path=None,
        v_bias_path=None,
    ):
        super().__init__()
        self.scaling = config.lora_alpha / config.lora_r
        self.lora_dropout = nn.Dropout(config.lora_dropout)

        self.q_proj_lora_A = nn.Parameter(
            torch.randn(config.hidden_size, config.lora_r, dtype=DTYPE)
        )
        nn.init.kaiming_uniform_(self.q_proj_lora_A, nonlinearity="linear")
        self.q_proj_lora_B = nn.Parameter(
            torch.zeros(
                config.lora_r, config.num_attention_heads * head_dim, dtype=DTYPE
            )
        )
        self.k_proj_lora_A = nn.Parameter(
            torch.randn(config.hidden_size, config.lora_r, dtype=DTYPE)
        )
        nn.init.kaiming_uniform_(self.k_proj_lora_A, nonlinearity="linear")
        self.k_proj_lora_B = nn.Parameter(
            torch.zeros(
                config.lora_r, config.num_key_value_heads * head_dim, dtype=DTYPE
            )
        )
        self.v_proj_lora_A = nn.Parameter(
            torch.randn(config.hidden_size, config.lora_r, dtype=DTYPE)
        )
        nn.init.kaiming_uniform_(self.v_proj_lora_A, nonlinearity="linear")
        self.v_proj_lora_B = nn.Parameter(
            torch.zeros(
                config.lora_r, config.num_key_value_heads * head_dim, dtype=DTYPE
            )
        )

        self.q_weight_path = q_weight_path
        self.k_weight_path = k_weight_path
        self.v_weight_path = v_weight_path

        self.q_proj_bias = None
        self.k_proj_bias = None
        self.v_proj_bias = None

        q_dim = config.num_attention_heads * head_dim
        kv_dim = config.num_key_value_heads * head_dim

        if q_bias_path is not None:
            self.q_proj_bias = load_tensor_from_storage(
                weight_path=q_bias_path, shape=(q_dim,), to_ram=True
            )

        if k_bias_path is not None:
            self.k_proj_bias = load_tensor_from_storage(
                weight_path=k_bias_path, shape=(kv_dim,), to_ram=True
            )

        if v_bias_path is not None:
            self.v_proj_bias = load_tensor_from_storage(
                weight_path=v_bias_path, shape=(kv_dim,), to_ram=True
            )

    def forward(self, x):
        x_dropped = self.lora_dropout(x)
        return CactusLoraQKVLinearFunction.apply(
            x_dropped,
            self.q_weight_path,
            self.k_weight_path,
            self.v_weight_path,
            self.q_proj_bias,
            self.k_proj_bias,
            self.v_proj_bias,
            self.q_proj_lora_A,
            self.q_proj_lora_B,
            self.k_proj_lora_A,
            self.k_proj_lora_B,
            self.v_proj_lora_A,
            self.v_proj_lora_B,
            self.scaling,
        )


class CactusAttention(nn.Module):
    def __init__(self, config: CactusConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj_weight_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.self_attn.q_proj.weight.bin"
        )
        self.k_proj_weight_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.self_attn.k_proj.weight.bin"
        )
        self.v_proj_weight_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.self_attn.v_proj.weight.bin"
        )
        self.q_proj_bias_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.self_attn.q_proj.bias.bin"
        )
        self.k_proj_bias_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.self_attn.k_proj.bias.bin"
        )
        self.v_proj_bias_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.self_attn.v_proj.bias.bin"
        )
        self.o_proj_weight_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.self_attn.o_proj.weight.bin"
        )

        self.lora = CactusLoraQKVLinear(
            config,
            self.head_dim,
            self.q_proj_weight_path,
            self.k_proj_weight_path,
            self.v_proj_weight_path,
            self.q_proj_bias_path,
            self.k_proj_bias_path,
            self.v_proj_bias_path,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, key_states, value_states = self.lora(hidden_states)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attn_output, attn_weights = self.cactus_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        with torch.no_grad():
            o_proj_weight = load_tensor_from_storage(
                weight_path=self.o_proj_weight_path,
                shape=(self.config.hidden_size, self.config.hidden_size),
                to_ram=False,
            )
        attn_output = CactusMatmulFunction.apply(attn_output, o_proj_weight.t(), 1.0)
        gc.collect()
        return attn_output, attn_weights

    def cactus_attention_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = self.repeat_kv(key, module.num_key_value_groups)
        value_states = self.repeat_kv(value, module.num_key_value_groups)

        attn_weights = CactusMatmulFunction.apply(
            query, key_states.transpose(2, 3), scaling
        )
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = CactusMatmulFunction.apply(attn_weights, value_states, 1.0)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, queries, keys, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        queries_embed = (queries * cos) + (self.rotate_half(queries) * sin)
        keys_embed = (keys * cos) + (self.rotate_half(keys) * sin)
        return queries_embed, keys_embed
