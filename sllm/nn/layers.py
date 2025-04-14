"""
This module implements various neural network layers and functions that are used for
lazy weight loading and efficient computation in a transformer-based model. The layers include:

    - Embedding: Token embedding that lazily loads weights from disk.
    - RotaryEmbedding: Implements rotary position embeddings with dynamic frequency updates.
    - RMSNorm: Root-mean-square layer normalization with lazy weight loading.
    - Linear: A linear layer that lazily loads large weight matrices.
    - MLP: Feed-forward neural network layer using bundled matrix multiplications.
    - LoraLinear: Linear layer with LoRA adaptation for efficient fine-tuning.
    - LoraQKVLinear: Specialized LoRA linear layer for query/key/value projections.
    - Attention: Multi-head attention layer with support for rotary embeddings and caching.

Each layer leverages lazy weight loading via the `load_tensor_from_storage` utility to
optimize memory usage.
"""

import gc
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sllm.common import DTYPE
from sllm.config import Config
from sllm.nn.autodiff import (BundledMatmulFunction,
                                LoraFunction,
                                LoraQKVLinearFunction,
                                MatmulFunction)
from sllm.utils import load_tensor_from_storage


class Embedding(torch.nn.Module):
    """
    Token embedding layer with lazy weight loading.

    This module loads the weight matrix from storage on each forward pass,
    avoiding the need to hold the full embedding matrix in RAM.
    """

    def __init__(self, weight_path, vocab_size, hidden_size, padding_idx=None):
        """
        Initialize the Embedding layer.

        Args:
            weight_path (str): Path to the weight file.
            vocab_size (int): Number of tokens in the vocabulary.
            hidden_size (int): Dimensionality of the embedding vectors.
            padding_idx (optional): Padding index for tokens; defaults to None.
        """
        super().__init__()
        self.weight_path = weight_path
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        """
        Perform a forward pass to retrieve embeddings for the provided token IDs.

        Args:
            input_ids (Tensor): Tensor containing token IDs with shape (...).

        Returns:
            Tensor: Embedding vectors corresponding to the input IDs.
        """
        with torch.no_grad():
            weight = load_tensor_from_storage(
                weight_path=self.weight_path,
                shape=(self.vocab_size, self.hidden_size),
                to_ram=False,
            )
        return weight[input_ids]


class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding module with dynamic frequency updates.

    Computes rotary embeddings and applies attention scaling. It updates the frequency
    parameters if the input sequence length exceeds cached values.
    """

    def __init__(self, config: Config):
        """
        Initialize the RotaryEmbedding module.

        Args:
            config (Config): Model configuration containing parameters such as
                max_position_embeddings and rope_theta.
        """
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        inv_freq, self.attention_scaling = self.compute_rope_parameters(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        Dynamically update the frequency parameters if needed.

        The update occurs if:
            1. The sequence length exceeds the cached maximum.
            2. The current sequence length is smaller than the original max sequence length
               while the cached maximum is larger.

        Args:
            position_ids (Tensor): Tensor of position indices.
            device (torch.device): The device on which to perform the computation.
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
        """
        Compute rotary embeddings for the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, ...).
            position_ids (Tensor): Position indices for each token.

        Returns:
            Tuple[Tensor, Tensor]: A tuple (cos, sin) where each tensor contains
            the cosine and sine components of the rotary embeddings.
        """
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

        # Apply attention scaling for advanced RoPE types (e.g., yarn)
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def compute_rope_parameters(
        self,
        config: Optional[Config] = None,
    ) -> Tuple["torch.Tensor", float]:
        """
        Compute the rotary embedding parameters.

        This method calculates the inverse frequency tensor and returns the corresponding attention scaling
        factor.

        Args:
            config (Optional[Config], optional): The model configuration. If not provided,
                defaults to the instance's configuration.

        Returns:
            Tuple[Tensor, float]: A tuple (inv_freq, attention_factor) where:
                - inv_freq is the inverse frequency tensor.
                - attention_factor is the scaling factor for attention.
        """
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


class RMSNorm(torch.nn.Module):
    """
    Root-mean-square layer normalization with lazy-loaded weights.

    Applies RMS normalization to the input tensor using pre-loaded weights.
    """

    def __init__(self, hidden_size, weight_path, eps=1e-6):
        """
        Initialize the RMSNorm layer.

        Args:
            hidden_size (int or tuple): The shape of the weight tensor.
            weight_path (str): Path to the weight file.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        weight = load_tensor_from_storage(
            weight_path=weight_path, shape=hidden_size, to_ram=True
        )
        self.register_buffer("weight", weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Apply RMS normalization to the input tensor.

        Args:
            hidden_states (Tensor): Input tensor with shape (..., hidden_size).

        Returns:
            Tensor: The normalized tensor scaled by the registered weight.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        """
        Return additional string representation of the RMSNorm module.

        Returns:
            str: A string representation showing the weight shape and epsilon value.
        """
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Linear(nn.Module):
    """
    Linear layer that lazily loads a large weight matrix from disk.

    The weight matrix is loaded on every forward pass with gradients disabled.
    The bias, if provided, is loaded entirely into RAM.
    """

    def __init__(self, in_features, out_features, weight_path, bias_path=None):
        """
        Initialize the Linear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            weight_path (str): Path to the weight file.
            bias_path (str, optional): Path to the bias file. Defaults to None.
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
        """
        Perform the forward pass of the linear layer.

        Loads the weight matrix from disk and computes the matrix multiplication.
        If bias is present, adds it to the result.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features).
        """
        weight = load_tensor_from_storage(
            weight_path=self.weight_path,
            shape=(self.out_features, self.in_features),
            to_ram=False,
        ).t()

        Wx = MatmulFunction.apply(x, weight, 1.0)
        if self.bias is not None:
            Wx += self.bias
        return Wx


class MLP(nn.Module):
    """
    Feed-forward network (MLP) module using bundled matrix multiplications.

    This layer implements a gated MLP where the input is projected with two parallel
    linear layers, one of which is passed through a non-linearity and then elementwise
    multiplied, followed by a final projection.
    """

    def __init__(self, config, layer_idx):
        """
        Initialize the MLP module.

        Args:
            config (Config): Model configuration including hidden and intermediate sizes.
            layer_idx (int): Index of the current layer for loading weight files.
        """
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
        """
        Compute the forward pass of the MLP.

        Projects the input using two parallel linear transformations (gate and up projections),
        applies a non-linearity and elementwise multiplication, and finally projects back down.

        Args:
            x (Tensor): Input tensor with shape (..., hidden_size).

        Returns:
            Tensor: Output tensor with shape (..., hidden_size).
        """
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
        gate_proj_out, up_proj_out = BundledMatmulFunction.apply(bundles)
        activated_gate_proj = self.act_fn(gate_proj_out) * up_proj_out
        return MatmulFunction.apply(activated_gate_proj, down_proj.t(), 1.0)


class LoraLinear(nn.Module):
    """
    LoRA adapted linear layer for efficient fine-tuning.

    This module applies LoRA by incorporating trainable low-rank matrices (lora_A, lora_B)
    to adjust the pre-trained weight matrix.
    """

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
        """
        Initialize the LoraLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            r (int): LoRA rank.
            alpha (int): LoRA scaling factor.
            weight_path (str): Path to the pre-trained weight file.
            bias_path (str, optional): Path to the bias file. Defaults to None.
            lora_dropout (float, optional): Dropout probability for LoRA. Defaults to 0.0.
        """
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
        """
        Forward pass for the LoraLinear layer.

        Applies dropout to the input, then uses the LoRA function to compute the modified linear transformation.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features).
        """
        x_dropped = self.lora_dropout(x)
        return LoraFunction.apply(
            x_dropped,
            self.lora_A,
            self.lora_B,
            self.weight_path,
            self.scaling,
            self.bias,
        )


class LoraQKVLinear(nn.Module):
    """
    LoRA adapted linear layer for query, key, and value projections in attention.

    This layer applies LoRA to the query, key, and value projections and lazily loads
    the corresponding weight matrices.
    """

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
        """
        Initialize the LoraQKVLinear layer.

        Args:
            config (Config): Model configuration with LoRA parameters.
            head_dim (int): Dimension of each attention head.
            q_weight_path (str): Path to the query projection weight file.
            k_weight_path (str): Path to the key projection weight file.
            v_weight_path (str): Path to the value projection weight file.
            q_bias_path (str, optional): Path to the query projection bias file.
            k_bias_path (str, optional): Path to the key projection bias file.
            v_bias_path (str, optional): Path to the value projection bias file.
        """
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
        """
        Forward pass for the LoRA QKV linear layer.

        Applies dropout to the input and computes the LoRA adapted query, key, and value projections.

        Args:
            x (Tensor): Input tensor of shape (..., hidden_size).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the projected query, key, and value tensors.
        """
        x_dropped = self.lora_dropout(x)
        return LoraQKVLinearFunction.apply(
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


class Attention(nn.Module):
    """
    Multi-head attention layer with support for rotary embeddings and caching.

    This layer computes queries, keys, and values using a LoRA adapted linear layer,
    applies rotary position embeddings, and then performs scaled dot-product attention.
    It also supports updating caches via past key/value mechanisms.
    """

    def __init__(self, config: Config, layer_idx: int):
        """
        Initialize the Attention layer.

        Args:
            config (Config): Model configuration.
            layer_idx (int): Index of the current layer for loading layer-specific weights.
        """
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

        self.lora = LoraQKVLinear(
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
        """
        Forward pass for the Attention layer.

        Projects inputs into queries, keys, and values, applies rotary embeddings, and computes
        the attention output along with optional attention weights.

        Args:
            hidden_states (Tensor): Input tensor with shape (batch_size, seq_length, hidden_size).
            position_embeddings (Tuple[Tensor, Tensor]): Tuple containing cosine and sine embeddings.
            attention_mask (Optional[Tensor]): Attention mask to apply.
            past_key_value (Optional[Any]): Cached past key/value states.
            cache_position (Optional[LongTensor]): Cache positions for tokens.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple:
                - attn_output (Tensor): Output tensor after attention.
                - attn_weights (Optional[Tensor]): Attention weights if computed.
        """
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

        attn_output, attn_weights = self.attention_forward(
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
        attn_output = MatmulFunction.apply(attn_output, o_proj_weight.t(), 1.0)
        gc.collect()
        return attn_output, attn_weights

    def attention_forward(
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
        """
        Compute the core attention mechanism.

        Applies scaled dot-product attention over the queries, keys, and values, taking into account
        the attention mask and dropout.

        Args:
            module (nn.Module): Reference to the current module (used for configuration).
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
            attention_mask (Optional[Tensor]): Mask to apply to the attention weights.
            scaling (float): Scaling factor applied to the dot product.
            dropout (float, optional): Dropout probability for the attention weights.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple:
                - attn_output (Tensor): The attention output tensor.
                - attn_weights (Tensor): The computed attention weights.
        """
        key_states = self.repeat_kv(key, module.num_key_value_groups)
        value_states = self.repeat_kv(value, module.num_key_value_groups)

        attn_weights = MatmulFunction.apply(
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
        attn_output = MatmulFunction.apply(attn_weights, value_states, 1.0)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key/value hidden states to match the required number of groups.

        Args:
            hidden_states (Tensor): Hidden states tensor with shape (batch, num_heads, seq_len, head_dim).
            n_rep (int): Number of repetitions.

        Returns:
            Tensor: Reshaped tensor with repeated key/value states.
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def rotate_half(self, x):
        """
        Rotate half of the tensor for rotary embedding application.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Tensor with rotated halves.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, queries, keys, cos, sin, unsqueeze_dim=1):
        """
        Apply rotary positional embeddings to queries and keys.

        Args:
            queries (Tensor): Query tensor.
            keys (Tensor): Key tensor.
            cos (Tensor): Cosine embedding.
            sin (Tensor): Sine embedding.
            unsqueeze_dim (int, optional): Dimension along which to unsqueeze the cosine and sine embeddings.
                Defaults to 1.

        Returns:
            Tuple[Tensor, Tensor]: Rotated queries and keys with positional embeddings applied.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        queries_embed = (queries * cos) + (self.rotate_half(queries) * sin)
        keys_embed = (keys * cos) + (self.rotate_half(keys) * sin)
        return queries_embed, keys_embed
