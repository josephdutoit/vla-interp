import os
import torch
import pickle
import argparse
from PIL import Image, ImageDraw
from rv_train.dataset import LiberoDataset
from tqdm import tqdm

def visualize_features(args):
    print(f"Loading features from {args.input_path}...")
    with open(args.input_path, "rb") as f:
        data = pickle.load(f)
    
    top_k_data = data["top_k"]
    feature_stats = data["feature_stats"]
    
    # Calculate average activation for each feature to find "important" ones
    avg_act = feature_stats["total_act"] / (feature_stats["count"] + 1e-8)
    # Sort by total activation or average activation? 
    # Usually total activation is good to find frequent strong features.
    important_features = torch.argsort(feature_stats["total_act"], descending=True)[:args.num_features]
    
    print(f"Loading dataset {args.repo_id}...")
    dataset = LiberoDataset(
        repo_id=args.repo_id,
        img_size=args.img_size,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for f_idx in tqdm(important_features.tolist(), desc="Visualizing features"):
        if f_idx not in top_k_data:
            continue
            
        f_dir = os.path.join(args.output_dir, f"feature_{f_idx}")
        os.makedirs(f_dir, exist_ok=True)
        
        examples = top_k_data[f_idx]
        
        # Create a summary image for this feature
        for i, (act, info) in enumerate(examples):
            d_idx = info["dataset_idx"]
            sample = dataset[d_idx]
            img = sample["images"][0] # PIL Image
            
            draw = ImageDraw.Draw(img)
            
            if info["is_vision"]:
                # Map vision_rel_idx to spatial location
                rel_idx = info["vision_rel_idx"]
                thw = info["grid_thw"]
                t, h, w = thw
                
                # Qwen2.5-VL uses a patch_size. For 224x224, it's usually 16x16 grid if patch_size=14
                # Let's calculate based on thw
                h_idx = (rel_idx // w) % h
                w_idx = rel_idx % w
                
                patch_h = img.height // h
                patch_w = img.width // w
                
                top = h_idx * patch_h
                left = w_idx * patch_w
                bottom = (h_idx + 1) * patch_h
                right = (w_idx + 1) * patch_w
                
                # Draw a red rectangle around the patch
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
            
            # Save the image
            img_path = os.path.join(f_dir, f"example_{i}_act_{act:.2f}.png")
            img.save(img_path)
            
            # Save metadata
            with open(os.path.join(f_dir, f"example_{i}_meta.txt"), "w") as mf:
                mf.write(f"Activation: {act:.4f}\n")
                mf.write(f"Instruction: {info['instruction']}\n")
                if not info["is_vision"]:
                    mf.write(f"Token: {info['token_text']}\n")
                else:
                    mf.write(f"Token: Vision Patch {info['vision_rel_idx']}\n")

    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/sae_features.pkl")
    parser.add_argument("--output_dir", type=str, default="data/feature_viz")
    parser.add_argument("--repo_id", type=str, default="HuggingFaceVLA/libero")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_features", type=int, default=20)
    
    args = parser.parse_args()
    visualize_features(args)
