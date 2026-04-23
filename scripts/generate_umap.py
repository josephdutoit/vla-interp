import os
import pickle
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm

def load_shards_for_umap(input_dir):
    """Loads all shards and prepares a [D_SAE, WorldSize * TopK] matrix."""
    shard_dir = os.path.join(input_dir, "shards")
    world_size = 8
    
    # We will build a matrix where each row is a feature's signature
    all_vals_list = []
    
    print(f"Loading 8 shards to build activation signatures...")
    for r in tqdm(range(world_size), desc="Loading Shards"):
        path = os.path.join(shard_dir, f"shard_{r}_final.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            # data['local_top_vals'] is [65536, 10]
            # Convert to CPU/Numpy immediately
            vals = data['local_top_vals'].numpy()
            all_vals_list.append(vals)
            
    # Concatenate columns: [65536, 80] 
    # This vector represents "how strong this feature was across the whole dataset"
    signature_matrix = np.concatenate(all_vals_list, axis=1)
    return signature_matrix

def main(args):
    # 1. Load Data
    X = load_shards_for_umap(args.input_dir)
    print(f"Signature matrix shape: {X.shape}") # [65536, 80]

    # 2. Filter out dead features
    # UMAP hates rows of all zeros
    row_sums = np.sum(np.abs(X), axis=1)
    active_mask = row_sums > 1e-6
    X_active = X[active_mask]
    active_ids = np.where(active_mask)[0]
    
    print(f"Removed {len(X) - len(X_active)} dead features. {len(X_active)} active features remain.")

    # 3. Standardize
    print("Standardizing data...")
    X_scaled = StandardScaler().fit_transform(X_active)

    # 4. Optional PCA Pre-processing (Speeds up UMAP and denoises)
    print("Running PCA pre-reduction (to 50 dims)...")
    pca = PCA(n_components=min(50, X_active.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # 5. Run UMAP
    # metric='cosine' is usually better for activations than 'euclidean'
    print("Starting UMAP projection (this may take 5-10 minutes)...")
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        metric='cosine', 
        verbose=True,
        random_state=42
    )
    embedding = reducer.fit_transform(X_pca)

    # 6. Save results
    output_data = {
        "f_ids": active_ids,
        "embedding": embedding,
        "max_acts": np.max(X_active, axis=1)
    }
    
    output_path = os.path.join(args.input_dir, "umap_embeddings.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)
    
    print(f"Success! UMAP coordinates saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with the 'shards' folder")
    args = parser.parse_args()
    main(args)