"""
This module implements a Transformer-based language model with support for decoder layers,
gradient checkpointing, and caching mechanisms. It defines the following classes:

    - DecoderLayer: Implements a single layer of the transformer decoder.
    - Transformer: Composes multiple decoder layers into a full transformer model.
    - SuperLazyLanguageModel: Encapsulates the transformer and head for sequence generation tasks.

The implementation leverages PyTorch for tensor computations and Hugging Face transformers
for cache management.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)

from sllm.common import DTYPE
from sllm.config import Config
from sllm.nn.layers import (Attention, Embedding, Linear,
                              MLP, RMSNorm, RotaryEmbedding)


class DecoderLayer(nn.Module):
    """
    A single decoder layer of the Transformer.

    This layer consists of:
        - RMS normalization applied to the input (input_layernorm).
        - A self-attention mechanism (self_attn).
        - A residual connection adding the result back to the input.
        - A second RMS normalization (post_attention_layernorm) followed by a feed-forward MLP,
          with an additional residual connection.

    Args:
        config (Config): Configuration object containing model hyperparameters.
        layer_idx (int): Index of the layer to load layer-specific weights.
    """

    def __init__(self, config: Config, layer_idx: int):
        """
        Initialize the decoder layer.

        Loads layer-specific weights for normalization from the provided weight directory.

        Args:
            config (Config): Model configuration.
            layer_idx (int): Index for the current layer.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(config, layer_idx)

        input_layer_norm_weight_path = (
            f"{config.weight_dir}/model.layers.{layer_idx}.input_layernorm.weight.bin"
        )
        post_attention_layer_norm_weight_path = f"{config.weight_dir}/model.layers.{layer_idx}.post_attention_layernorm.weight.bin"

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            weight_path=input_layer_norm_weight_path,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            weight_path=post_attention_layer_norm_weight_path,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Perform a forward pass through the decoder layer.

        The input passes through an initial layer normalization, self-attention block, and
        a feed-forward MLP with residual connections. Optionally, attention weights can be returned.

        Args:
            hidden_states (torch.Tensor): Input tensor with shape (batch_size, seq_length, hidden_size).
            attention_mask (Optional[torch.Tensor], optional): Attention mask for self-attention.
            position_ids (Optional[torch.LongTensor], optional): Tensor containing position indices.
            past_key_value (Optional[Cache], optional): Cached past key and value tensors.
            output_attentions (Optional[bool], optional): If True, returns self-attention weights.
            use_cache (Optional[bool], optional): If True, enables caching for inference.
            cache_position (Optional[torch.LongTensor], optional): Positions for caching tokens.
            position_embeddings (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): Pre-computed position embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
                - The output tensor of shape (batch_size, seq_length, hidden_size).
                - Optionally, a tuple of self-attention weights.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Transformer(nn.Module):
    """
    Transformer decoder composed of multiple decoder layers.

    This module includes:
        - Token embedding.
        - A stack of decoder layers.
        - Rotary positional embeddings.
        - Final normalization.
        - Optional support for gradient checkpointing and caching.

    Attributes:
        padding_idx (int): Padding index for token embeddings.
        vocab_size (int): Size of the vocabulary.
        embed_tokens (Embedding): Token embedding layer.
        layers (nn.ModuleList): A list of decoder layers.
        norm (RMSNorm): Normalization applied after the last decoder layer.
        rotary_emb (RotaryEmbedding): Module for rotary position embeddings.
        gradient_checkpointing (bool): Enables gradient checkpointing if True.
        config (Config): Model configuration.
    """

    def __init__(self, config: Config):
        """
        Initialize the Transformer decoder.

        Loads weights for token embeddings, each decoder layer, and the final normalization.

        Args:
            config (Config): Configuration object with model hyperparameters.
        """
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            padding_idx=self.padding_idx,
            weight_path=f"{config.weight_dir}/model.embed_tokens.weight.bin",
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        norm_weight_path = f"{config.weight_dir}/model.norm.weight.bin"
        self.norm = RMSNorm(
            config.hidden_size, weight_path=norm_weight_path, eps=config.rms_norm_eps
        )
        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.config = config

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings.

        Returns:
            Embedding: The embedding layer used for token lookup.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings.

        Args:
            value (Embedding): A new embedding layer.
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Perform a forward pass through the Transformer decoder.

        Delegates embedding lookup to the embedding layer, applies rotary position embeddings,
        and passes data sequentially through each decoder layer. Also manages caching and generates
        an updated causal mask based on past key values.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs.
            attention_mask (Optional[torch.Tensor], optional): Attention mask for the sequence.
            position_ids (Optional[torch.LongTensor], optional): Position IDs for the sequence.
            past_key_values (Optional[Cache], optional): Cached key/value pairs.
            inputs_embeds (Optional[torch.FloatTensor], optional): Pre-computed input embeddings.
            use_cache (Optional[bool], optional): If True, enables caching.
            output_attentions (Optional[bool], optional): If True, outputs attention weights.
            output_hidden_states (Optional[bool], optional): If True, outputs hidden states.
            return_dict (Optional[bool], optional): If True, returns a dict-like object instead of a tuple.
            cache_position (Optional[torch.LongTensor], optional): Cache position indices.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                - If return_dict is False, returns a tuple with logits and optionally additional outputs.
                - Otherwise, returns a `BaseModelOutputWithPast` with last hidden state, cached key values,
                  hidden states, and attentions.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: dict,
    ):
        """
        Generate a causal attention mask accounting for cached tokens.

        The mask is created based on the current input tensor, the cache positions, and the type of caching
        used (static or sliding window). This allows the model to correctly attend to previous tokens and
        manage attention when using caches.

        Args:
            attention_mask (torch.Tensor): Original attention mask.
            input_tensor (torch.Tensor): Input tensor with shape (batch_size, seq_length, hidden_size).
            cache_position (torch.Tensor): Tensor indicating positions for cached tokens.
            past_key_values (dict): Cached past key values, possibly an instance of StaticCache or SlidingWindowCache.

        Returns:
            torch.Tensor: A 4D causal attention mask.
        """
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Config,
        past_key_values: dict,
    ):
        """
        Create a 4D causal attention mask with cache positions.

        This function prepares a mask of shape (batch_size, 1, query_length, key_value_length) from a
        2D mask or creates a new one if needed. For sliding window configurations, additional masking is applied.

        Args:
            attention_mask (torch.Tensor): Input attention mask.
            sequence_length (int): Length of the current sequence.
            target_length (int): The target length (e.g., including cached tokens).
            dtype (torch.dtype): Data type for the mask.
            device (torch.device): Device for the mask tensor.
            cache_position (torch.Tensor): Tensor indicating positions for caching.
            batch_size (int): Batch size.
            config (Config): Model configuration (may include sliding window settings).
            past_key_values (dict): Cached past key values.

        Returns:
            torch.Tensor: A 4D causal attention mask tensor of shape (batch_size, 1, query_length, key_value_length).
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask

        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)

            if config.sliding_window is not None:
                if (
                    not isinstance(past_key_values, SlidingWindowCache)
                    or sequence_length > target_length
                ):
                    sliding_attend_mask = torch.arange(
                        target_length, device=device
                    ) <= (cache_position.reshape(-1, 1) - config.sliding_window)
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask


class SuperLazyLanguageModel(nn.Module):
    """
    A super lazy language model that encapsulates a Transformer decoder with optional LoRA parameters.

    This model wraps the Transformer decoder, defines the head for vocabulary logits,
    and computes the loss when labels are provided. It supports caching for efficient generation.

    Attributes:
        config (Config): Model configuration containing hyperparameters.
        model (Transformer): Underlying Transformer-based decoder.
        loss_function (nn.CrossEntropyLoss): Loss function for training.
        vocab_size (int): Size of the model vocabulary.
        lm_head (Linear): Linear projection layer from hidden states to vocabulary logits.
    """

    def __init__(self, name, lora_alpha=16, lora_r=4, lora_dropout=0.1):
        """
        Initialize the SuperLazyLanguageModel.

        Loads configuration and initializes the transformer decoder along with the language modeling head.
        Optionally, applies LoRA modifications if enabled in the configuration.

        Args:
            name (str): Identifier or name of the model.
            lora_alpha (int, optional): LoRA alpha hyperparameter. Defaults to 16.
            lora_r (int, optional): LoRA rank. Defaults to 4.
            lora_dropout (float, optional): LoRA dropout rate. Defaults to 0.1.
        """
        super().__init__()

        self.config = Config(
            model_name=name,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
            lora_dropout=lora_dropout,
        )

        self.model = Transformer(self.config)
        self.loss_function = nn.CrossEntropyLoss()
        self.vocab_size = self.config.vocab_size

        if self.config.tie_word_embeddings:
            lm_head_weight_path = (
                f"{self.config.weight_dir}/model.embed_tokens.weight.bin"
            )
        else:
            lm_head_weight_path = f"{self.config.weight_dir}/lm_head.weight.bin"

        self.lm_head = Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            weight_path=lm_head_weight_path,
            bias_path=None,
        )

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the model.

        Returns:
            Embedding: Input embedding layer.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the model.

        Args:
            value (Embedding): New input embedding layer.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Retrieve the output embeddings (language modeling head).

        Returns:
            Linear: The linear layer projecting to vocabulary logits.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings (language modeling head) for the model.

        Args:
            new_embeddings (Linear): New output embedding layer.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Replace the current Transformer decoder with a new one.

        Args:
            decoder (Transformer): A new transformer decoder module.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Retrieve the current Transformer decoder.

        Returns:
            Transformer: The underlying transformer decoder.
        """
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Perform a forward pass through the SuperLazyLanguageModel.

        The method computes embeddings (or uses precomputed ones), obtains outputs from the Transformer decoder,
        projects hidden states to vocabulary logits using the lm_head, and computes the cross-entropy loss if labels are provided.
        It supports caching for efficient autoregressive generation and can return outputs as a tuple or as a dict-like object.

        Args:
            input_ids (torch.LongTensor, optional): Token IDs as input.
            attention_mask (Optional[torch.Tensor], optional): Mask to avoid attending to padding tokens.
            position_ids (Optional[torch.LongTensor], optional): Position IDs corresponding to the tokens.
            past_key_values (Optional[Union[Cache, List[torch.FloatTensor]]], optional): Cached key/value tensors.
            inputs_embeds (Optional[torch.FloatTensor], optional): Pre-computed input embeddings.
            labels (Optional[torch.LongTensor], optional): Ground truth labels for computing the loss.
            use_cache (Optional[bool], optional): Whether to use past key values for caching.
            output_attentions (Optional[bool], optional): Whether to return attention weights.
            output_hidden_states (Optional[bool], optional): Whether to return hidden states.
            return_dict (Optional[bool], optional): Whether to return a dict-like output.
            cache_position (Optional[torch.LongTensor], optional): Cache positions for tokens.
            logits_to_keep (Union[int, torch.Tensor], optional): Slicing index or tensor for logits computation.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]:
                - If return_dict is False, returns a tuple containing logits and optionally other outputs.
                - Otherwise, returns a `CausalLMOutputWithPast` with loss (if computed), logits, past key values,
                  hidden states, and attention weights.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            loss = self.loss_function(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
