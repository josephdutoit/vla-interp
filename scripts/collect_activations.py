import os
import sys
import torch
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))
from rv_train.model import QwenVLActor
from rv_train.dataset import LiberoDataset
from rv_train.collator import VLACollator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_checkpoint(output_dir):
    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            return json.load(f)
    return {"samples_processed": 0, "total_tokens": 0, "chunk_idx": 0}

def save_checkpoint(output_dir, checkpoint):
    ckpt_path = os.path.join(output_dir, "checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=4)

def collect_activations(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = get_checkpoint(args.output_dir)
    print(f"Resuming from sample index {checkpoint['samples_processed']}, total tokens {checkpoint['total_tokens']}")
    
    if checkpoint["total_tokens"] >= args.num_samples:
        print(f"Target of {args.num_samples} tokens already reached.")
        return

    # Load model and processor
    model = QwenVLActor(
        model_path=args.model_path,
        stats_path=args.stats_path,
        torch_compile=False,
    )
    
    processor = model.processor
    
    # Load dataset
    dataset = LiberoDataset(
        repo_id=args.repo_id,
        img_size=args.img_size,
    )

    print(f"Dataset size: {len(dataset)} samples")
    
    # Deterministic shuffling of dataset indices
    all_indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(all_indices)
    
    # Resume by slicing the shuffled indices
    remaining_indices = all_indices[checkpoint["samples_processed"]:]
    print(f"Dataset size: {len(all_indices)}. Remaining samples: {len(remaining_indices)}")
    
    if not remaining_indices:
        print("No more samples to process in the dataset.")
        return

    subset = Subset(dataset, remaining_indices)
    
    # Use the subset for collection. shuffle=False because we already shuffled indices.
    dataloader = DataLoader(
        subset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=VLACollator(processor),
        num_workers=4
    )
    
    activations_buffer = []
    tokens_in_buffer = 0
    samples_in_buffer = 0
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations_buffer.append(output.detach().cpu())

    try:
        target_layer = model.model.model.language_model.layers[args.layer_idx]
    except AttributeError:
        target_layer = model.model.layers[args.layer_idx]
        
    handle = target_layer.register_forward_hook(hook_fn)
    
    pbar = tqdm(total=len(subset), initial=checkpoint["total_tokens"])
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # if checkpoint["total_tokens"] >= args.num_samples:
            #     break
            
            # Prepare inputs
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            _ = model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask
            )
            
            # Extract activations from hook
            batch_acts = activations_buffer.pop()
            mask = attention_mask.cpu().bool()
            filtered_acts = batch_acts[mask]
            
            activations_buffer.append(filtered_acts)
            new_tokens = filtered_acts.shape[0]
            tokens_in_buffer += new_tokens
            checkpoint["total_tokens"] += new_tokens
            samples_in_buffer += input_ids.shape[0]
            pbar.update(input_ids.shape[0])
            
            # Save chunk to disk if buffer exceeds chunk_size
            if tokens_in_buffer >= args.chunk_size:
                chunk_data = torch.cat(activations_buffer, dim=0)
                save_path = os.path.join(args.output_dir, f"activations_chunk_{checkpoint['chunk_idx']}.pt")
                torch.save(chunk_data, save_path)
                
                # Update checkpoint
                checkpoint["samples_processed"] += samples_in_buffer
                checkpoint["chunk_idx"] += 1
                save_checkpoint(args.output_dir, checkpoint)
                
                # Reset buffers
                activations_buffer = []
                tokens_in_buffer = 0
                samples_in_buffer = 0

    handle.remove()
    pbar.close()
    
    # Save any remaining activations in the final chunk
    if activations_buffer:
        chunk_data = torch.cat(activations_buffer, dim=0)
        save_path = os.path.join(args.output_dir, f"activations_chunk_{checkpoint['chunk_idx']}.pt")
        torch.save(chunk_data, save_path)
        
        checkpoint["samples_processed"] += samples_in_buffer
        checkpoint["chunk_idx"] += 1
        save_checkpoint(args.output_dir, checkpoint)
        print(f"Saved final chunk. Collection complete for this session.")

    print(f"Total tokens collected: {checkpoint['total_tokens']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='../logs/runs/vla0-trl-test', help="Path to trained model checkpoint")
    parser.add_argument(
        "--stats_path", type=str, default=None, help="Path to dataset_stats.json (auto-detected if not specified)"
    )
    parser.add_argument("--repo_id", type=str, default="HuggingFaceVLA/libero")
    parser.add_argument("--layer_idx", type=int, default=11)
    parser.add_argument("--num_samples", type=int, default=100000000000)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="data/activations_l11")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    collect_activations(args)
