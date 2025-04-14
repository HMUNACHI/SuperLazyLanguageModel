"""
This module defines the configuration class for the transformer-based model.
It encapsulates all the hyperparameters and weight configurations required for the model
and handles downloading and merging with the pretrained configuration from Hugging Face.

Key functionalities include:
    - Setting model hyperparameters such as hidden size, number of layers, attention heads, etc.
    - Configuring LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning.
    - Downloading necessary weights for the specified model.
    - Overwriting default parameters with those from the pretrained configuration.
"""

import os

from transformers import AutoConfig

from sllm.common import WEIGHT_DIR
from sllm.utils import download_weights, remove_weights


class Config:
    """
    A configuration object for initializing the transformer-based model.

    This class holds various hyperparameters required by the model. It downloads pretrained
    weights if needed and loads configurations from a pretrained model, overriding any matching 
    attributes provided in the constructor. In addition to standard model settings, it also 
    includes configuration details for LoRA adaptation.

    Class Attributes:
        model_type (str): Default type of the model, set to "transformer".
        keys_to_ignore_at_inference (List[str]): List of keys (e.g., "past_key_values") to be 
            ignored during inference.
    """

    model_type = "transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        vocab_size=151936,
        bos_token_id=151643,
        eos_token_id=151643,
        model_type="qwen2",
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_mrope=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=21,
        attention_dropout=0.0,
        torch_dtype="bfloat16",
        pad_token_id=None,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        lora_alpha=32,
        lora_r=8,
        lora_dropout=0.1,
        **kwargs,
    ):
        """
        Initialize the Config object with the provided hyperparameters and then update with the pretrained configuration.

        The initialization process involves:
            1. Setting initial values for many model hyperparameters and LoRA parameters.
            2. Determining the local weight directory based on the provided model name.
            3. Downloading the pretrained weights (if not already present) into the specified directory.
            4. Loading additional configuration parameters from the pretrained model via AutoConfig,
               and updating the current configuration for any matching attributes.

        Side Effects:
            - Downloads the pretrained weights (if not already available) to a local weight directory.
            - Updates attributes of this Config instance based on the pretrained model's configuration.
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        self.use_mrope = use_mrope
        self.torch_dtype = torch_dtype
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_type = model_type
        self.pad_token_id = pad_token_id
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout

        # Determine local weight storage directory based on the model name.
        model_dir = model_name.split("/")[-1]
        self.weight_dir = f"{WEIGHT_DIR}/{model_dir}"

        # Ensure that the pretrained weights are downloaded locally.
        download_weights(self.weight_dir, model_name)

        # Load additional configuration parameters from the pretrained model.
        config = AutoConfig.from_pretrained(model_name, **kwargs)

        # Overwrite any matching attributes in this Config instance with the pretrained values.
        for key, value in config.__dict__.items():
            if hasattr(self, key):
                setattr(self, key, value)
