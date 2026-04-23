import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from rv_train.model import load_model_for_training, load_processor
from rv_train.dataset import LiberoDataset
from rv_train.collator import VLACollator
from rv_interp.models.sae import SparseAutoencoder
import numpy as np

def eval_sae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    model = load_model_for_training(args.model_id, use_flash_attention=True)
    model.to(device)
    model.eval()
    
    processor = load_processor(args.model_id)
    
    # Load SAE
    d_model = 2048 # Standard for Qwen2.5-VL-3B
    d_sae = d_model * args.expansion_factor
    sae = SparseAutoencoder(d_model, d_sae)
    sae.load_state_dict(torch.load(args.sae_path, map_location=device))
    sae.to(device)
    sae.eval()
    
    # Load dataset
    dataset = LiberoDataset(
        repo_id=args.repo_id,
        img_size=args.img_size,
    )
    
    collator = VLACollator(processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collator
    )
    
    activations_list = []
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations_list.append(output.detach())

    try:
        target_layer = model.model.language_model.layers[args.layer_idx]
    except AttributeError:
        target_layer = model.model.layers[args.layer_idx]
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    total_mse = 0
    total_l0 = 0
    total_ev = 0
    total_tokens = 0
    
    # To store feature activations for analysis
    # We'll just look at a few samples
    print("Evaluating SAE...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= args.num_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask
            )
            
            batch_acts = activations_list.pop() # (batch, seq_len, d_model)
            
            # Filter non-pad tokens
            mask = attention_mask.bool()
            x = batch_acts[mask].to(torch.float32)
            
            output = sae(x)
            
            total_mse += output["mse_loss"].item() * x.shape[0]
            total_l0 += output["l0"].item() * x.shape[0]
            total_ev += output["explained_variance"].item() * x.shape[0]
            total_tokens += x.shape[0]

    handle.remove()
    
    avg_mse = total_mse / total_tokens
    avg_l0 = total_l0 / total_tokens
    avg_ev = total_ev / total_tokens
    
    print("\nEvaluation Results:")
    print(f"Total Tokens: {total_tokens}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average L0 (Sparsity): {avg_l0:.2f}")
    print(f"Explained Variance: {avg_ev:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--repo_id", type=str, default="physical-intelligence/libero")
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=11)
    parser.add_argument("--expansion_factor", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=224)
    
    args = parser.parse_args()
    eval_sae(args)
