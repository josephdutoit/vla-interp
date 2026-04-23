import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import numpy as np
from tqdm import tqdm
import pickle
from datetime import timedelta

# Local source imports
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))
from rv_train.model import QwenVLActor
from rv_train.dataset import LiberoDataset
from rv_train.collator import VLACollator
from rv_interp.models.sae import SparseAutoencoder

def setup_dist():
    """Initializes DDP with a massive timeout to prevent 'Silent Death' at 97%."""
    dist.init_process_group(
        backend="nccl", 
        init_method="env://",
        timeout=timedelta(hours=4) # 4 hours to handle even the slowest rank merge
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def save_shard(rank, output_dir, data, is_final=True):
    """Utility to save shards immediately to disk."""
    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    
    suffix = "final" if is_final else "checkpoint"
    path = os.path.join(shard_dir, f"shard_{rank}_{suffix}.pkl")
    
    with open(path, "wb") as f:
        pickle.dump(data, f)
        f.flush()
        os.fsync(f.fileno()) # Force write to BYU cluster storage
    print(f"Rank {rank} saved {suffix} shard: {path}", flush=True)

def collect_shards(args):
    local_rank = setup_dist()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    # 1. Setup Models
    model_actor = QwenVLActor(model_path=args.model_path, stats_path=args.stats_path)
    model = model_actor.model.to(device).eval()

    d_model = 2048
    d_sae = d_model * args.expansion_factor
    sae = SparseAutoencoder(d_model, d_sae).to(device).eval()
    sae.load_state_dict(torch.load(args.sae_path, map_location=device))

    # 2. Dataset & Sharded Loader
    dataset = LiberoDataset(repo_id=args.repo_id, img_size=args.img_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Increased num_workers to speed up image processing bottleneck
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=8, 
        pin_memory=True,
        collate_fn=VLACollator(model_actor.processor, action_mask_aug_pct=0.0)
    )

    # 3. Buffers
    local_top_vals = torch.zeros((d_sae, args.top_k), device=device)
    local_top_idxs = torch.full((d_sae, args.top_k), -1, dtype=torch.long, device=device)
    
    # New buffers for top-k unique instructions
    local_top_instruct_vals = torch.zeros((d_sae, args.top_k), device=device)
    local_top_instruct_idxs = torch.full((d_sae, args.top_k), -1, dtype=torch.long, device=device)
    local_top_instruct_instr_ids = torch.full((d_sae, args.top_k), -1, dtype=torch.long, device=device)
    
    all_instructions = []
    instr_to_id = {}
    local_sample_instructions = [] # To map sample index to instruction
    
    local_token_counts = []
    local_token_offset = 0

    # 4. Hooks
    activations_list = []
    def hook_fn(mod, inp, out):
        activations_list.append(out[0] if isinstance(out, tuple) else out)

    try:
        target_layer = model.model.language_model.layers[args.layer_idx]
    except AttributeError:
        target_layer = model.layers[args.layer_idx]
    handle = target_layer.register_forward_hook(hook_fn)

    print(f"Rank {rank} starting collection loop...", flush=True)
    
    # 5. Main Loop
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).bool()
            instructions = batch["instructions"]
            
            _ = model(
                input_ids=input_ids,
                pixel_values=batch["pixel_values"].to(device),
                image_grid_thw=batch["image_grid_thw"].to(device),
                attention_mask=attention_mask
            )
            
            batch_acts = activations_list.pop()
            
            for b in range(input_ids.shape[0]):
                instr = instructions[b]
                if instr not in instr_to_id:
                    instr_to_id[instr] = len(all_instructions)
                    all_instructions.append(instr)
                instr_id = instr_to_id[instr]
                local_sample_instructions.append(instr)

                sample_acts = batch_acts[b][attention_mask[b]].to(torch.float32)
                num_tokens = sample_acts.shape[0]
                local_token_counts.append(num_tokens)
                
                sae_acts = sae.encoder(sample_acts)
                
                # Global top_k
                active_f = torch.where(sae_acts.max(dim=0).values > 0)[0]
                
                # Instruction-wise top_k
                sample_max_vals, sample_max_token_idxs = sae_acts.max(dim=0)
                
                for f_id in active_f:
                    # 1. Update Global Top-K
                    f_vals = sae_acts[:, f_id]
                    combined_vals = torch.cat([local_top_vals[f_id], f_vals])
                    
                    new_indices = torch.arange(num_tokens, device=device) + local_token_offset
                    combined_idxs = torch.cat([local_top_idxs[f_id], new_indices])
                    
                    best_vals, best_sort_idx = torch.topk(combined_vals, args.top_k)
                    local_top_vals[f_id] = best_vals
                    local_top_idxs[f_id] = combined_idxs[best_sort_idx]

                    # 2. Update Top-K Unique Instructions
                    val = sample_max_vals[f_id]
                    idx = sample_max_token_idxs[f_id] + local_token_offset
                    
                    existing_instr_ids = local_top_instruct_instr_ids[f_id]
                    match = (existing_instr_ids == instr_id).nonzero()
                    
                    if match.numel() > 0:
                        p = match[0].item()
                        if val > local_top_instruct_vals[f_id, p]:
                            local_top_instruct_vals[f_id, p] = val
                            local_top_instruct_idxs[f_id, p] = idx
                    else:
                        min_val, min_p = torch.min(local_top_instruct_vals[f_id], dim=0)
                        if val > min_val:
                            local_top_instruct_vals[f_id, min_p] = val
                            local_top_instruct_idxs[f_id, min_p] = idx
                            local_top_instruct_instr_ids[f_id, min_p] = instr_id
                
                local_token_offset += num_tokens

            # Progress Logging (Every 50 batches)
            if rank == 0 and i % 50 == 0:
                print(f"[PROGRESS] Batch {i}/{len(dataloader)} | Shard Tokens: {local_token_offset:,}", flush=True)

            # --- INSURANCE: SAVE CHECKPOINT EVERY 1000 BATCHES ---
            if i > 0 and i % 1000 == 0:
                ckpt_data = {
                    "local_top_vals": local_top_vals.cpu(),
                    "local_top_idxs": local_top_idxs.cpu(),
                    "local_top_instruct_vals": local_top_instruct_vals.cpu(),
                    "local_top_instruct_idxs": local_top_instruct_idxs.cpu(),
                    "local_top_instruct_instr_ids": local_top_instruct_instr_ids.cpu(),
                    "all_instructions": all_instructions,
                    "local_sample_instructions": local_sample_instructions,
                    "local_token_counts": local_token_counts,
                    "batch_idx": i
                }
                save_shard(rank, args.output_dir, ckpt_data, is_final=False)

    handle.remove()

    # 6. Final Save (NO dist.barrier() BEFORE SAVING)
    final_data = {
        "local_top_vals": local_top_vals.cpu(),
        "local_top_idxs": local_top_idxs.cpu(),
        "local_top_instruct_vals": local_top_instruct_vals.cpu(),
        "local_top_instruct_idxs": local_top_instruct_idxs.cpu(),
        "local_top_instruct_instr_ids": local_top_instruct_instr_ids.cpu(),
        "all_instructions": all_instructions,
        "local_sample_instructions": local_sample_instructions,
        "local_token_counts": local_token_counts,
        "rank": rank
    }
    save_shard(rank, args.output_dir, final_data, is_final=True)

    # Clean cleanup
    print(f"Rank {rank} finished successfully.", flush=True)
    dist.barrier() 
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='../logs/runs/vla0-trl-test', help="Path to trained model checkpoint")
    parser.add_argument(
        "--stats_path", type=str, default=None, help="Path to dataset_stats.json (auto-detected if not specified)"
    )
    parser.add_argument("--sae_path", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="HuggingFaceVLA/libero")
    parser.add_argument("--expansion_factor", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--layer_idx", type=int, default=11)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    
    args = parser.parse_args()
    collect_shards(args)