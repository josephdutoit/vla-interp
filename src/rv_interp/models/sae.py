import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_sae: int, l1_coeff: float = 3e-4):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder.bias.data.zero_()
        
        self.decoder = nn.Linear(d_sae, d_model)
        self.decoder.weight.data = self.encoder.weight.data.t().clone()
        self.decoder.bias.data.zero_()
        
        self.set_decoder_norm()

    def set_decoder_norm(self):
        with torch.no_grad():
            self.decoder.weight.data /= self.decoder.weight.data.norm(dim=0, keepdim=True)

    def forward(self, x):
        # x: (batch, d_model)
        x_centered = x - self.decoder.bias
        
        # Encode
        pre_acts = self.encoder(x_centered)
        acts = F.relu(pre_acts)
        
        # Decode
        x_reconstruct = self.decoder(acts) + self.decoder.bias
        
        # Loss
        mse_loss = F.mse_loss(x_reconstruct, x)
        l1_loss = acts.abs().sum(dim=-1).mean()
        loss = mse_loss + self.l1_coeff * l1_loss
        
        # Metrics
        with torch.no_grad():
            # L0: number of active features per sample
            l0 = (acts > 0).float().sum(dim=-1).mean()
            
            # Explained Variance
            per_token_var = (x - x.mean(dim=0)).pow(2).sum()
            resid_var = (x - x_reconstruct).pow(2).sum()
            explained_variance = 1 - resid_var / (per_token_var + 1e-8)
        
        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "l0": l0,
            "explained_variance": explained_variance,
            "acts": acts,
            "x_reconstruct": x_reconstruct
        }

    @torch.no_grad()
    def make_decoder_unit_norm(self):
        self.decoder.weight.data /= self.decoder.weight.data.norm(dim=0, keepdim=True)
