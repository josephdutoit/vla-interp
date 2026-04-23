import sys
import pickle
import pandas as pd
from enum import Enum
from tqdm import tqdm
import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Update your paths
sys.path.insert(0, "src")
from rv_train.dataset import LiberoDataset
from rv_train.model import load_processor
from qwen_vl_utils import process_vision_info

EMBEDDINGS_PATH = "data/discovered_features/layer_25_2/umap_embeddings.pkl"
FEATURES_PATH = "data/discovered_features/layer_25_2/global_features.pkl"
MODEL_PATH = "logs/runs/vla0-trl-test"
IMAGE_PAD_ID = 151655

class FeatureClass(Enum):
    VISION = 0
    LANGUAGE = 1
    ACTION = 2
    MULTIMODAL = 3

# Global variables for worker processes
worker_processor = None
worker_dataset = None

def init_worker():
    """ 
    Runs once per CPU core. Loads the heavy objects into the worker's 
    local memory space so they don't have to be re-loaded for every feature.
    """
    global worker_processor, worker_dataset
    # Load processor (usually light enough to replicate across 128 cores)
    worker_processor = load_processor(MODEL_PATH, img_size=224, num_cams=2, tile_images=True)
    # Dataset is usually just a file manifest, so it's safe to replicate
    worker_dataset = LiberoDataset(repo_id="HuggingFaceVLA/libero", img_size=224)

def classify_feature_chunk(feature_data_chunk):
    """
    Processes a batch of features. 
    feature_data_chunk is a list of tuples: (feature_id, list_of_examples)
    """
    chunk_results = {}
    
    for fid, examples in feature_data_chunk:
        current_class = None
        
        for ex in examples:
            # 1. Prepare inputs
            sample = worker_dataset[ex['sample_idx']]
            messages = sample['messages']
            images = sample['images']
            
            # Inject images into message structure
            img_idx = 0
            for msg in messages:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content["type"] == "image":
                            content["image"] = images[img_idx]
                            img_idx += 1
            
            # 2. Tokenization (Heavy CPU Work)
            text = worker_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            inputs = worker_processor(text=[text], images=image_inputs, return_tensors="pt")
            
            input_ids = inputs['input_ids'][0]
            token_idx = ex["token_idx"]
            
            # 3. Classification Logic
            activated_token_id = input_ids[token_idx].item()
            
            point_class = None
            if activated_token_id == IMAGE_PAD_ID:
                point_class = FeatureClass.VISION
            else:
                # Check for Action (Numeric) vs Language
                start_idx = max(0, token_idx - 5)
                end_idx = min(len(input_ids), token_idx + 5)
                decoded = worker_processor.tokenizer.decode(input_ids[start_idx:end_idx])
                whitespace_removed = "".join(decoded.split())
                decoded_max_act = worker_processor.tokenizer.decode(input_ids[token_idx])
                # Need to make this part
                if whitespace_removed.isnumeric() or \
                    (("assistant" in decoded.lower() or "im_end" in decoded.lower()) and decoded_max_act.isnumeric()):
                    point_class = FeatureClass.ACTION
                else:
                    point_class = FeatureClass.LANGUAGE
            
            # 4. Aggregate Multimodal logic
            if current_class is None:
                current_class = point_class
            elif current_class != point_class:
                current_class = FeatureClass.MULTIMODAL
        
        chunk_results[fid] = current_class
        
    return chunk_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-workers', type=int, default=128)
    args = parser.parse_args()

    print("Loading Master Data...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        umap_data = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        global_results = pickle.load(f)

    feature_ids = umap_data['f_ids'].tolist()
    
    # Bundle IDs and their examples together into CHUNKS
    # This reduces the number of messages sent between processes
    print(f"Dividing {len(feature_ids)} features into {args.num_workers} chunks...")
    feature_items = [(fid, global_results.get(fid, [])) for fid in feature_ids]

    n = args.num_workers
    chunks = [feature_items[i::n] for i in range(n)]

    results = {}
    
    # ProcessPoolExecutor manages the 128 workers
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker
    ) as executor:
        
        # Submit each chunk as one job
        futures = [executor.submit(classify_feature_chunk, chunk) for chunk in chunks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
            results.update(future.result())

    # Map results back to the original UMAP dataframe
    df = pd.DataFrame({
        'x': umap_data['embedding'][:, 0],
        'y': umap_data['embedding'][:, 1],
        'f_id': umap_data['f_ids'],
        'max_act': umap_data['max_acts']
    })
    
    # Store the result as strings/names for easier use in Dash
    df['feature_class'] = df['f_id'].map(lambda fid: results.get(fid).name if results.get(fid) else "UNKNOWN")

    output_path = "data/discovered_features/classified_features.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(df, f)
    
    print(f"Successfully saved {len(df)} classified features to {output_path}")

if __name__ == "__main__":
    main()