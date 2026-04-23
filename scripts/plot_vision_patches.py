import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from rv_train.model import load_processor
from rv_train.dataset import LiberoDataset

def plot_patches_on_image(sample_idx, patch_indices, model_path, output_path="patch_plot.png"):
    """
    Loads a frame and plots specific vision patch indices on it.
    
    Args:
        sample_idx: Index of the sample in the dataset.
        patch_indices: List of relative vision token indices (0 to num_patches-1).
        model_path: Path to the model/processor.
        output_path: Where to save the resulting image.
    """
    print(f"Loading processor from {model_path}...")
    # Using parameters consistent with the visualization app
    processor = load_processor(model_path, img_size=224, num_cams=2, tile_images=True)
    
    print("Loading dataset...")
    dataset = LiberoDataset(
        repo_id="HuggingFaceVLA/libero", 
        img_size=224,
        crop_ratio=1.0,
        brightness_aug=0,
        contrast_aug=0,
        saturation_aug=0,
        hue_aug=0
    )
    
    # Get the sample
    sample = dataset[sample_idx]
    images = sample['images']  # List of PIL Images
    messages = sample['messages']
    
    print(f"Processing frame {sample_idx}...")
    
    # Inject images into messages for process_vision_info
    img_idx = 0
    for msg in messages:
        if isinstance(msg["content"], list):
            for content in msg["content"]:
                if content["type"] == "image":
                    content["image"] = images[img_idx]
                    img_idx += 1

    # Apply template and process vision info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt")
    
    # Qwen2-VL specific grid calculation
    # The first image (index 0) is usually the tiled image if tile_images=True
    display_image = images[0].copy()
    
    if "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"][0] # [T, H, W]
        # Qwen2-VL tokens are 2x2 pooled versions of the original patches
        h_grid, w_grid = grid_thw[1].item() // 2, grid_thw[2].item() // 2
        
        print(f"Image Grid: {h_grid}x{w_grid} ({h_grid * w_grid} total vision tokens)")
        print(f"Image Dimensions: {display_image.width}x{display_image.height}")
        
        draw = ImageDraw.Draw(display_image, "RGBA")
        patch_h = display_image.height / h_grid
        patch_w = display_image.width / w_grid
        
        for rel_idx in patch_indices:
            if rel_idx >= h_grid * w_grid:
                print(f"Warning: Patch index {rel_idx} is out of bounds for grid {h_grid}x{w_grid}")
                continue
                
            row_idx = rel_idx // w_grid
            col_idx = rel_idx % w_grid
            
            left = col_idx * patch_w
            top = row_idx * patch_h
            right = left + patch_w
            bottom = top + patch_h
            
            # Draw a semi-transparent red box with a thick border
            draw.rectangle([left, top, right, bottom], fill=(255, 0, 0, 80), outline=(255, 0, 0, 255), width=2)
            
            # Add text label for the index
            draw.text((left + 2, top + 2), str(rel_idx), fill=(255, 255, 255, 255))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(display_image)
        ax.set_title(f"Activated Vision Patches (sample {sample_idx})")
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"Plot saved to {output_path}")
    else:
        print("Error: image_grid_thw not found in processor inputs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot specific vision patches on a VLA frame.")
    parser.add_argument("--idx", type=int, default=1000, help="Dataset sample index")
    parser.add_argument("--patches", type=int, nargs="+", default=[0, 10, 50, 100], help="List of vision token indices")
    parser.add_argument("--model_path", type=str, default="logs/runs/vla0-trl-test")
    parser.add_argument("--output", type=str, default="vision_patches_1000.png")
    
    args = parser.parse_args()
    
    plot_patches_on_image(args.idx, args.patches, args.model_path, args.output)
