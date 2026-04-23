#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))
from rv_interp.data.corrupted_dataset import CorruptedLiberoDataset
from rv_train.model import QwenVLActor
from qwen_vl_utils import process_vision_info


VISION_START_TAG = '<|vision_start|>'
VISION_END_TAG = '<|vision_end|>'

def find_token_categories(processor, input_ids):
    """
    Categorize tokens into: system, vision, instruction, action.
    Ensures contiguous regions covering all tokens.
    """
    im_start_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    assistant_id = processor.tokenizer.convert_tokens_to_ids('assistant')
    
    ids_list = input_ids.tolist()
    categories = {"system": [], "vision": [], "instruction": [], "action": []}
    
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
            categories["vision"].append(i)
        elif assistant_start != -1 and i >= assistant_start:
            categories["action"].append(i)
        elif v_end != -1 and i > v_end and i < assistant_start:
            categories["instruction"].append(i)
        else:
            # Before vision starts (or before assistant starts if no vision)
            # This covers the system prompt and the user start markers before vision
            categories["system"].append(i)
            
    return categories

def add_modality_markers(ax, system_indices, vision_indices, instruction_indices, action_indices, is_y=True, is_x=True):
    """Add lines and labels to indicate modalities on the heatmap."""
    modalities = [
        ("System", system_indices, "gray"),
        ("Vision", vision_indices, "red"),
        ("Instr", instruction_indices, "blue"),
        ("Action", action_indices, "green")
    ]
    
    for name, indices, color in modalities:
        if not indices: continue
        start, end = min(indices), max(indices)
        
        # Adjust by 0.5 to enclose the pixels (which are centered on integers)
        start_f = start - 0.5
        end_f = end + 0.5
        
        if is_x:
            ax.axvline(x=start_f, color=color, linestyle='-', alpha=0.8, lw=1.5)
            ax.axvline(x=end_f, color=color, linestyle='-', alpha=0.8, lw=1.5)
            # Add text label at the top
            ax.text((start_f + end_f) / 2, -5, name, color=color, ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        if is_y:
            ax.axhline(y=start_f, color=color, linestyle='-', alpha=0.8, lw=1.5)
            ax.axhline(y=end_f, color=color, linestyle='-', alpha=0.8, lw=1.5)
            # Add text label at the left
            ax.text(-5, (start_f + end_f) / 2, name, color=color, ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

def get_attention_maps(model, inputs):
    """
    Run forward pass and return attention maps.
    """
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    return outputs.attentions # Tuple of (batch, heads, seq_len, seq_len)

def plot_attention_grid(attentions, system_indices, vision_indices, instruction_indices, action_indices, save_path, title):
    """
    Plot a grid of attention heatmaps or summary statistics.
    Since full NxN is too big, let's plot:
    X-axis: Source tokens (Vision, Instruction, Action)
    Y-axis: Layers
    Value: Mean attention from action tokens to source tokens.
    """
    num_layers = len(attentions)
    
    # We want to see where the ACTION tokens are looking.
    # Action tokens are typically the ones being generated or the last ones in the prompt.
    
    results = {
        "System": [],
        "Vision": [],
        "Instruction": [],
        "Past Actions": [],
        "Other": []
    }
    
    for layer_attn in attentions:
        # layer_attn: (1, num_heads, seq_len, seq_len)
        mean_attn = layer_attn[0].mean(dim=0).to(torch.float32) # (seq_len, seq_len)
        
        # Attention from Action tokens to others
        if action_indices:
            attn_from_actions = mean_attn[action_indices, :] # (len(actions), seq_len)
            
            results["System"].append(attn_from_actions[:, system_indices].mean().item())
            results["Vision"].append(attn_from_actions[:, vision_indices].mean().item())
            results["Instruction"].append(attn_from_actions[:, instruction_indices].mean().item())
            results["Past Actions"].append(attn_from_actions[:, action_indices].mean().item())
            
            all_special = set(system_indices) | set(vision_indices) | set(instruction_indices) | set(action_indices)
            other_indices = [i for i in range(mean_attn.size(1)) if i not in all_special]
            if other_indices:
                results["Other"].append(attn_from_actions[:, other_indices].mean().item())
            else:
                results["Other"].append(0.0)
        else:
            for k in results: results[k].append(0.0)

    plt.figure(figsize=(12, 6))
    layers = range(num_layers)
    for label, values in results.items():
        plt.plot(layers, values, marker='o', label=label)
        
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Mean Attention Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def save_attention_plots(attn_map, save_dir, prefix, title_prefix, system_indices, vision_indices, instruction_indices, action_indices, instruction_tokens):
    """Save various attention plots for a given attention map (e.g., a specific head or layer average)."""
    plot_attn = attn_map
    
    # Full sequence attention heatmap (all-to-all)
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    im = ax.imshow(plot_attn, cmap="viridis", aspect="auto", interpolation='nearest', vmin=0.0, vmax=1.0)
    plt.colorbar(im, label='Attention Weight')
    
    add_modality_markers(ax, system_indices, vision_indices, instruction_indices, action_indices)
    
    plt.suptitle(f"{title_prefix} Full Sequence Attention Map")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_dir, f"{prefix}_full_attention.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Close-up: Instruction attending to Vision
    if instruction_indices and vision_indices:
        plt.figure(figsize=(20, max(4, 0.3 * len(instruction_indices))))
        subset_attn = plot_attn[instruction_indices, :][:, vision_indices]
        ax = sns.heatmap(subset_attn, cmap="viridis", vmin=0.0, vmax=1.0,
                         yticklabels=instruction_tokens, cbar_kws={'label': 'Attention Weight'})
        plt.title(f"{title_prefix} Close-up: Instruction tokens attending to Vision tokens")
        plt.xlabel("Vision Token Index (within block)")
        plt.ylabel("Instruction Token")
        plt.savefig(os.path.join(save_dir, f"{prefix}_instr_to_vision.png"), bbox_inches='tight')
        plt.close()

    # Close-up: Actions attending to Instruction
    if action_indices and instruction_indices:
        plt.figure(figsize=(max(8, 0.3 * len(instruction_indices)), 8))
        subset_attn = plot_attn[action_indices, :][:, instruction_indices]
        ax = sns.heatmap(subset_attn, cmap="viridis", vmin=0.0, vmax=1.0,
                         xticklabels=instruction_tokens, cbar_kws={'label': 'Attention Weight'})
        plt.title(f"{title_prefix} Close-up: Action tokens attending to Instruction tokens")
        plt.xlabel("Instruction Token")
        plt.ylabel("Action Token Index (within block)")
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(save_dir, f"{prefix}_action_to_instr.png"), bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='../logs/runs/vla0-trl-test')
    parser.add_argument("--output_dir", type=str, default='data/attention_outputs')
    parser.add_argument("--num_frames", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model_wrapper = QwenVLActor(model_path=args.model_path, attn_implementation="eager")
    model = model_wrapper.model
    processor = model_wrapper.processor

    print("Loading dataset...")
    dataset = CorruptedLiberoDataset(
        repo_id="HuggingFaceVLA/libero",
        corrupted_instructions_map="../../data/corrupted-data/corrupted_objects.json"
    )

    # Use tracer's logic to find indices
    from rv_interp.analysis.causal_tracer import CausalTracer
    tracer = CausalTracer(model_wrapper)

    # indices = random.sample(range(len(dataset)), args.num_frames)
    indices = [3000]
    
    for i, idx in enumerate(tqdm(indices, desc="Processing frames")):
        data = dataset[idx]
        
        text = processor.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(data["messages"])
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"][0]
        
        # Identify indices
        all_categories = find_token_categories(processor, input_ids)
        system_indices = all_categories["system"]
        vision_indices = all_categories["vision"]
        instruction_indices = all_categories["instruction"]
        action_indices = all_categories["action"]
        
        # Decode instruction tokens for labeling
        instruction_tokens = [processor.tokenizer.decode([input_ids[j]]) for j in instruction_indices]
        
        # Get attentions
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)
            attentions = outputs.attentions # Tuple of (batch, heads, seq_len, seq_len)
            
            if attentions is None:
                print(f"Warning: No attentions returned for frame {idx}")
                continue
                            
        # Create sub-directory for this frame
        frame_dir = os.path.join(args.output_dir, f"frame_{idx:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        
        # 1. Plot aggregate attention curve
        plot_attention_grid(
            attentions, 
            system_indices,
            vision_indices, 
            instruction_indices, 
            action_indices,
            os.path.join(frame_dir, "attention_summary.png"),
            f"Attention Summary - Frame {idx}\n{data['messages'][1]['content'][1]['text']}"
        )
        
        # 2. Plot heatmaps for each layer and head
        for layer_idx in tqdm(range(len(attentions)), desc=f"Plotting layers for frame {idx}", leave=False):
            layer_dir = os.path.join(frame_dir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_dir, exist_ok=True)
            
            # layer_attn: (1, num_heads, seq_len, seq_len)
            layer_attn = attentions[layer_idx][0]
            num_heads = layer_attn.shape[0]
            
            # 2a. Plot average for the layer
            mean_attn = layer_attn.mean(dim=0).to(torch.float32).cpu().numpy()
            save_attention_plots(
                mean_attn, layer_dir, "avg", f"Layer {layer_idx} (Avg)",
                system_indices, vision_indices, instruction_indices, action_indices, instruction_tokens
            )
            
            # 2b. Plot each head
            for head_idx in range(num_heads):
                head_attn = layer_attn[head_idx].to(torch.float32).cpu().numpy()
                save_attention_plots(
                    head_attn, layer_dir, f"head_{head_idx:02d}", f"Layer {layer_idx} Head {head_idx}",
                    system_indices, vision_indices, instruction_indices, action_indices, instruction_tokens
                )

    print(f"Done! Plots saved in {args.output_dir}")

if __name__ == "__main__":
    main()
