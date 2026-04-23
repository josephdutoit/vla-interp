import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from tqdm import tqdm
from rv_interp.models.sae import SparseAutoencoder
import glob
import wandb
import random

def get_activation_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.pt")))
        if not files:
            raise ValueError(f"No .pt files found in {input_path}")
        return files
    else:
        raise ValueError(f"Input path {input_path} not found")

def train_sae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get all activation files
    files = get_activation_files(args.input_path)
    print(f"Found {len(files)} activation chunks in {args.input_path}")
    
    # Load first chunk to get dimension
    print(f"Loading first chunk {files[0]} to determine dimension...")
    first_chunk = torch.load(files[0])
    d_model = first_chunk.shape[1]
    del first_chunk # Free memory
    
    if args.use_wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        wandb.init(
            project="sae-vla",
            name=args.run_name,
            config=vars(args),
            dir=args.wandb_dir,
            mode="offline"
        )
    
    # Initialize SAE
    d_sae = d_model * args.expansion_factor
    model = SparseAutoencoder(d_model, d_sae, l1_coeff=args.l1_coeff)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        
        # Shuffle chunks for this epoch
        random.shuffle(files)
        
        total_loss = 0
        total_mse = 0
        total_l1 = 0
        total_l0 = 0
        total_ev = 0
        num_batches = 0
        
        epoch_pbar = tqdm(files, desc=f"Epoch {epoch+1}/{args.epochs} (Chunks)")
        for chunk_idx, f in enumerate(epoch_pbar):
            # Load and convert to float32
            # Use weights from the file but don't keep it on GPU yet if large
            activations = torch.load(f).to(torch.float32)
            
            # Create DataLoader for this chunk
            dataset = TensorDataset(activations)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            
            chunk_pbar = tqdm(dataloader, desc=f"  Processing chunk {chunk_idx}", leave=False)
            for batch in chunk_pbar:
                x = batch[0].to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = output["loss"]
                loss.backward()
                
                # Constraint: decoder weights must be unit norm
                optimizer.step()
                model.make_decoder_unit_norm()
                
                # Metrics
                batch_loss = loss.item()
                batch_mse = output["mse_loss"].item()
                batch_l1 = output["l1_loss"].item()
                batch_l0 = output["l0"].item()
                batch_ev = output["explained_variance"].item()

                total_loss += batch_loss
                total_mse += batch_mse
                total_l1 += batch_l1
                total_l0 += batch_l0
                total_ev += batch_ev
                num_batches += 1
                
                chunk_pbar.set_postfix({
                    "mse": f"{batch_mse:.4f}",
                    "l0": f"{batch_l0:.1f}",
                    "ev": f"{batch_ev:.4f}"
                })

                if args.use_wandb:
                    wandb.log({
                        "batch/loss": batch_loss,
                        "batch/mse_loss": batch_mse,
                        "batch/l1_loss": batch_l1,
                        "batch/l0": batch_l0,
                        "batch/explained_variance": batch_ev,
                    }, step=global_step)
                
                global_step += 1
            
            # Update epoch progress bar with average metrics so far
            avg_loss_so_far = total_loss / num_batches
            avg_mse_so_far = total_mse / num_batches
            avg_l1_so_far = total_l1 / num_batches
            avg_l0_so_far = total_l0 / num_batches
            avg_ev_so_far = total_ev / num_batches
            
            epoch_pbar.set_postfix({
                "avg_mse": f"{avg_mse_so_far:.4f}",
                "avg_l0": f"{avg_l0_so_far:.1f}",
                "avg_ev": f"{avg_ev_so_far:.4f}"
            })

            # Clean up activations from memory
            del activations
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_l1 = total_l1 / num_batches
        avg_l0 = total_l0 / num_batches
        avg_ev = total_ev / num_batches
        
        print(f"Epoch {epoch+1} summary: Loss={avg_loss:.4f}, MSE={avg_mse:.4f}, L1={avg_l1:.4f}, L0={avg_l0:.1f}, EV={avg_ev:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/mse_loss": avg_mse,
                "epoch/l1_loss": avg_l1,
                "epoch/l0": avg_l0,
                "epoch/explained_variance": avg_ev,
                "epoch": epoch + 1
            }, step=global_step)

        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, f"sae_l{args.layer_idx}_e{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, f"sae_l{args.layer_idx}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../../data/activation_outputs/layer_25")
    parser.add_argument("--output_dir", type=str, default="../../data/sae_checkpoints")
    parser.add_argument("--layer_idx", type=int, default=25)
    parser.add_argument("--expansion_factor", type=int, default=32)
    parser.add_argument("--l1_coeff", type=float, default=3e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=10)
    
    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_dir", type=str, default="../../wandb")
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    train_sae(args)
