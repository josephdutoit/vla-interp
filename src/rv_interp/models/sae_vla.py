"""SAE-augmented Qwen2.5-VL Actor for feature steering."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from rv_train.model import QwenVLActor
from rv_interp.models.sae import SparseAutoencoder


class SAEQwenVLActor(QwenVLActor):
    """
    Extends QwenVLActor to include a Sparse Autoencoder (SAE) at a specific layer.
    Allows for observing and steering internal features by modifying the SAE bottleneck.
    """

    def __init__(
        self,
        model_path: str,
        sae_path: str,
        layer_idx: int = 11,
        *,
        stats_path: Optional[str] = None,
        horizon: int = 8,
        action_dim: int = 7,
        num_bins: int = 1000,
        device: str = "cuda",
        torch_compile: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        """
        Initializes the SAE-augmented actor.

        Args:
            model_path: Path to the Qwen2.5-VL model.
            sae_path: Path to the Sparse Autoencoder checkpoint.
            layer_idx: The index of the layer to intercept (default 11).
            stats_path: Path to action normalization statistics.
            horizon: Prediction horizon.
            action_dim: Action dimension.
            num_bins: Number of bins for action discretization.
            device: Device to load models on.
            torch_compile: Whether to use torch.compile on the LLM.
            attn_implementation: Attention implementation to use.
        """
        super().__init__(
            model_path,
            stats_path=stats_path,
            horizon=horizon,
            action_dim=action_dim,
            num_bins=num_bins,
            device=device,
            torch_compile=torch_compile,
            attn_implementation=attn_implementation,
        )

        self.layer_idx = layer_idx

        # Extract d_model from config
        if hasattr(self.model.config, "text_config"):
            self.d_model = self.model.config.text_config.hidden_size
        else:
            self.d_model = self.model.config.hidden_size

        # Load SAE
        checkpoint = torch.load(sae_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Infer d_sae from state_dict
        d_sae = None
        for key in ["encoder.weight", "module.encoder.weight"]:
            if key in state_dict:
                d_sae = state_dict[key].shape[0]
                break
        
        if d_sae is None:
            # Fallback for different naming conventions
            for key in state_dict.keys():
                if "encoder.weight" in key:
                    d_sae = state_dict[key].shape[0]
                    break
        
        if d_sae is None:
            raise ValueError(f"Could not infer SAE dimensions from checkpoint at {sae_path}")
            
        self.d_sae = d_sae
        self.sae = SparseAutoencoder(d_model=self.d_model, d_sae=self.d_sae)
        
        # Clean state dict (remove DDP prefix if present)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.sae.load_state_dict(clean_state_dict)
        self.sae.to(self.device)
        self.sae.eval()

        # Steering state
        self.steering_weights = None  # (d_sae,) tensor
        self.hook_handle = None
        
        # Register hook
        self._register_sae_hook()

    def _register_sae_hook(self):
        """Registers a forward hook on the target layer."""

        def hook_fn(module, input, output):
            # Qwen2-VL layer output is (hidden_states, ...)
            hidden_states = output[0]
            orig_dtype = hidden_states.dtype
            orig_shape = hidden_states.shape

            # Flatten to (batch * seq, d_model) and cast to float32 for SAE
            x = hidden_states.reshape(-1, self.d_model).to(torch.float32)

            with torch.no_grad():
                # SAE forward pass logic
                # 1. Center
                x_centered = x - self.sae.decoder.bias
                
                # 2. Encode
                pre_acts = self.sae.encoder(x_centered)
                acts = F.relu(pre_acts)
                
                # 3. Steering
                if self.steering_weights is not None:
                    # Apply multipliers to features
                    acts = acts * self.steering_weights
                
                # 4. Decode
                x_reconstruct = self.sae.decoder(acts) + self.sae.decoder.bias

            # Restore shape and dtype
            new_hidden_states = x_reconstruct.reshape(orig_shape).to(orig_dtype)
            
            # Return tuple to match original output signature
            return (new_hidden_states,) + output[1:]

        # Attach to the specific layer in the text model
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model") and hasattr(self.model.model.language_model, "layers"):
            target_layer = self.model.model.language_model.layers[self.layer_idx]
            self.hook_handle = target_layer.register_forward_hook(hook_fn)
        else:
            raise AttributeError("Could not find transformer layers in the model. Check model structure.")

    def set_steering_weights(self, weights: Optional[torch.Tensor]):
        """
        Sets weights to scale specific SAE features.
        
        Args:
            weights: Tensor of shape (d_sae,) or None to reset.
        """
        if weights is not None:
            if weights.shape != (self.d_sae,):
                raise ValueError(f"Weights shape {weights.shape} must match d_sae {self.d_sae}")
            self.steering_weights = weights.to(device=self.device, dtype=torch.float32)
        else:
            self.steering_weights = None

    def remove_hook(self):
        """Removes the intervention hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __del__(self):
        self.remove_hook()
