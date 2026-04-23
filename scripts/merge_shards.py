import os
import pickle
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))
from rv_train.dataset import LiberoDataset

def process_chunk(f_id_chunk, all_vals, all_idxs, all_counts, rank_indices, repo_id, img_size, top_k):
    """Worker function: processes a chunk of features and returns a dict."""
    local_results = {}
    world_size = 8
    
    # Initialize dataset locally in the worker to avoid pickling/shared-memory lock issues
    dataset = LiberoDataset(repo_id=repo_id, img_size=img_size)
    
    for f_id in f_id_chunk:
        # Concatenate 8 shards into one search space [8 * K]
        f_candidates_vals = np.concatenate([all_vals[r][f_id] for r in range(world_size)])
        
        # Get Global Top-K indices
        top_indices = np.argsort(f_candidates_vals)[-top_k:][::-1]
        final_vals = f_candidates_vals[top_indices]
        
        if final_vals[0] <= 1e-6:
            continue
            
        f_metadata = []
        for i, idx_in_concat in enumerate(top_indices):
            val = float(final_vals[i])
            if val <= 0: continue
            
            # Map back to which rank and k-slot this was
            r = idx_in_concat // top_k
            k_slot = idx_in_concat % top_k
            
            local_token_idx = int(all_idxs[r][f_id][k_slot])
            if local_token_idx == -1: continue
            
            # Shard-level math: find which sample this was in the 8,500
            offsets = np.cumsum([0] + all_counts[r])
            shard_sample_idx = np.searchsorted(offsets, local_token_idx, side='right') - 1
            token_pos = local_token_idx - offsets[shard_sample_idx]
            
            # Convert to global dataset index [0-68,000]
            global_sample_idx = rank_indices[r][shard_sample_idx]
            
            # Only pull instruction text if the feature is a 'winner'
            instr = dataset[global_sample_idx]["messages"][1]["content"][1]["text"]
            
            f_metadata.append({
                "value": val,
                "sample_idx": int(global_sample_idx),
                "token_idx": int(token_pos),
                "instruction": instr
            })
            
        local_results[int(f_id)] = f_metadata
    return local_results

def merge_shards_parallel(args):
    # 1. Load data once on the main process
    shard_dir = os.path.join(args.input_dir, "shards")
    world_size = 8
    all_vals, all_idxs, all_counts = [], [], []

    print(f"Loading {world_size} shards into main RAM...")
    for r in range(world_size):
        path = os.path.join(shard_dir, f"shard_{r}_final.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            # Move to numpy here so memory can be shared read-only with workers
            all_vals.append(data["local_top_vals"].numpy()) 
            all_idxs.append(data["local_top_idxs"].numpy())
            all_counts.append(data["local_token_counts"])

    # 2. Reconstruct Sampler Logic (Once)
    temp_ds = LiberoDataset(repo_id=args.repo_id, img_size=args.img_size)
    ds_len = len(temp_ds)
    indices = list(range(ds_len))
    padding_size = (world_size - ds_len % world_size) % world_size
    indices += indices[:padding_size]
    rank_indices = [np.array(indices[r::world_size]) for r in range(world_size)]
    del temp_ds 

    # 3. Define Parallel Chunks
    d_sae = all_vals[0].shape[0] # Usually 65536
    top_k = all_vals[0].shape[1]
    num_cpus = cpu_count()
    
    # We create slightly smaller chunks so the progress bar is smoother
    num_chunks = num_cpus * 4 
    f_id_chunks = np.array_split(range(d_sae), num_chunks)
    
    print(f"Merging 65k features using {num_cpus} cores (sharded into {num_chunks} tasks)...")
    
    worker_func = partial(
        process_chunk, 
        all_vals=all_vals, 
        all_idxs=all_idxs, 
        all_counts=all_counts, 
        rank_indices=rank_indices, 
        repo_id=args.repo_id,
        img_size=args.img_size,
        top_k=top_k
    )

    # 4. Multiprocessing with Progress Bar
    global_results = {}
    with Pool(processes=num_cpus) as pool:
        # imap_unordered lets us get results as they finish, rather than waiting for them in order
        for result_dict in tqdm(pool.imap_unordered(worker_func, f_id_chunks), total=len(f_id_chunks), desc="Global Merge"):
            global_results.update(result_dict)

    # 5. Save the final product
    output_path = os.path.join(args.input_dir, "global_features.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(global_results, f)
    
    print(f"Merge Complete! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="HuggingFaceVLA/libero")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    merge_shards_parallel(args)