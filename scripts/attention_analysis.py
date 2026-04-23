import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import json
import pandas as pd
import numpy as np
import time

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from rv_train.model import QwenVLActor
from rv_train.dataset import LiberoDataset
from rv_train.collator import VLACollator

# Modality constants
SYSTEM = 0
VISION = 1
INSTRUCTION = 2
ACTION = 3
MODALITY_NAMES = ["System", "Vision", "Instruction", "Action"]

class AttentionTracker:
    def __init__(self, num_layers):
        self.attentions = [None] * num_layers

    def hook_fn(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attentions[layer_idx] = attn_weights.detach().cpu()
        return hook

def find_modality_map(processor, input_ids):
    """
    Categorize tokens into: system, vision, instruction, action.
    Returns a tensor of the same shape as input_ids.
    """
    im_start_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    assistant_id = processor.tokenizer.convert_tokens_to_ids('assistant')
    
    ids_list = input_ids.tolist()
    modality = torch.full_like(input_ids, SYSTEM)
    
    # Find key transition points
    v_start = -1
    v_end = -1
    assistant_start = -1
    
    for i in range(len(ids_list)):
        if ids_list[i] == vision_start_id:
            v_start = i
        if ids_list[i] == vision_end_id:
            v_end = i
        if ids_list[i] == im_start_id and i + 1 < len(ids_list) and ids_list[i+1] == assistant_id:
            assistant_start = i
            
    # Assign every token to a category based on boundaries
    for i in range(len(ids_list)):
        if v_start != -1 and i >= v_start and i <= v_end:
            modality[i] = VISION
        elif assistant_start != -1 and i >= assistant_start:
            modality[i] = ACTION
        elif v_end != -1 and i > v_end and i < assistant_start:
            modality[i] = INSTRUCTION
        else:
            modality[i] = SYSTEM
            
    return modality

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="logs/runs/vla0-trl-test")
    parser.add_argument("--repo_id", type=str, default="HuggingFaceVLA/libero")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process. -1 for all.")
    parser.add_argument("--output_file", type=str, default="data/attention_density.json")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    # Get Rank/World Size from SLURM environment variables
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"Running PORT-FREE parallelization (Rank {rank}/{world_size})")
        print(f"Loading model from {args.model_path}...")
        
    # Load model
    model_actor = QwenVLActor(
        model_path=args.model_path, 
        device=f"cuda:{local_rank}",
        attn_implementation="eager"
    )
    model = model_actor.model
    processor = model_actor.processor
    model.config.output_attentions = True
    model.eval()

    # Identify layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model, "language_model"):
        layers = model.language_model.layers
    else:
        layers = model.layers
    num_layers = len(layers)
    
    # Register hooks
    tracker = AttentionTracker(num_layers)
    # Register hooks on self_attn specifically to get attention weights
    handles = [layer.self_attn.register_forward_hook(tracker.hook_fn(i)) for i, layer in enumerate(layers)]

    # Load and SHARD dataset manually
    dataset = LiberoDataset(repo_id=args.repo_id, img_size=args.img_size)
    total_len = len(dataset)
    if args.num_samples != -1 and args.num_samples < total_len:
        total_len = args.num_samples
    
    # Each rank takes a specific slice
    my_indices = list(range(rank, total_len, world_size))
    my_subset = Subset(dataset, my_indices)
    
    if rank == 0:
        print(f"Total samples: {total_len}. Rank 0 processing {len(my_subset)} samples.")

    dataloader = DataLoader(
        my_subset, batch_size=args.batch_size, collate_fn=VLACollator(processor), num_workers=2
    )

    # Local Accumulators (4x4 matrix for 4 modalities)
    density_acc = torch.zeros((num_layers, 4, 4), dtype=torch.float64, device=device)
    count_acc = torch.zeros((num_layers, 4), dtype=torch.float64, device=device)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=(rank != 0), desc="Analyzing")):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            model(
                input_ids=input_ids, pixel_values=pixel_values,
                image_grid_thw=image_grid_thw, attention_mask=attention_mask,
                output_attentions=True
            )
            
            for b in range(input_ids.shape[0]):
                v_mod = find_modality_map(processor, input_ids[b])
                v_mask = attention_mask[b] == 1
                
                # Apply mask to modality map
                v_mod = v_mod[v_mask]
                
                for l_idx in range(num_layers):
                    if tracker.attentions[l_idx] is None: continue
                    
                    # tracker.attentions[l_idx]: (batch, heads, seq_len, seq_len)
                    # Get attention for current batch element and apply mask
                    attn = tracker.attentions[l_idx][b].to(device).mean(dim=0)[v_mask][:, v_mask]
                    
                    for src in range(4):
                        src_indices = (v_mod == src).nonzero(as_tuple=True)[0]
                        if src_indices.numel() == 0: continue
                        
                        # Count how many query tokens of this modality exist
                        count_acc[l_idx, src] += src_indices.numel()
                        
                        for tgt in range(4):
                            tgt_indices = (v_mod == tgt).nonzero(as_tuple=True)[0]
                            if tgt_indices.numel() == 0: continue
                            
                            # Sum of attention from all tokens in src to all tokens in tgt
                            density_acc[l_idx, src, tgt] += attn[src_indices][:, tgt_indices].sum()

    # SAVE LOCAL RESULTS TO DISK
    tmp_dir = os.path.join(os.path.dirname(args.output_file), "tmp_attn")
    os.makedirs(tmp_dir, exist_ok=True)
    local_save_path = os.path.join(tmp_dir, f"rank_{rank}.pt")
    torch.save({"density": density_acc.cpu(), "count": count_acc.cpu()}, local_save_path)
    
    for h in handles: h.remove()

    # RANK 0 MERGES EVERYTHING
    if rank == 0:
        print("Waiting for all ranks to save results...")
        expected_files = [os.path.join(tmp_dir, f"rank_{i}.pt") for i in range(world_size)]
        
        # Wait until all files exist
        all_ready = False
        while not all_ready:
            all_ready = all(os.path.exists(f) for f in expected_files)
            if not all_ready: time.sleep(5)
        
        print("Merging results...")
        global_density = torch.zeros((num_layers, 4, 4), dtype=torch.float64)
        global_count = torch.zeros((num_layers, 4), dtype=torch.float64)
        
        for f in expected_files:
            data = torch.load(f)
            global_density += data["density"]
            global_count += data["count"]
            os.remove(f) # Clean up
            
        # Final compute
        final_results = []
        for l in range(num_layers):
            layer_data = {"layer": l}
            for i, f_mod in enumerate(MODALITY_NAMES):
                for j, t_mod in enumerate(MODALITY_NAMES):
                    val = 0
                    if global_count[l, i] > 0:
                        # Average attention per query token of modality i to all tokens of modality j
                        val = (global_density[l, i, j] / global_count[l, i]).item()
                    layer_data[f"{f_mod}_to_{t_mod}"] = val
            final_results.append(layer_data)
            
        with open(args.output_file, "w") as f:
            json.dump(final_results, f, indent=4)
        pd.DataFrame(final_results).to_csv(args.output_file.replace(".json", ".csv"), index=False)
        print(f"SUCCESS: Results saved to {args.output_file}")
        if os.path.exists(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)

if __name__ == "__main__":
    main()
