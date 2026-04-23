#!/usr/bin/env python3
from dataclasses import dataclass
import argparse, os, sys
import pandas as pd
import random

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))
from attrs import field
from tqdm import tqdm
from rv_interp.analysis.causal_tracer import CausalTracer
from rv_interp.data.corrupted_dataset import CorruptedLiberoDataset
from rv_train.model import QwenVLActor

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Tracing Experiment for VLA-0")
    parser.add_argument("--model_path", type=str, default='../logs/runs/vla0-trl-test', help="Path to trained model checkpoint")
    parser.add_argument(
        "--stats_path", type=str, default=None, help="Path to dataset_stats.json (auto-detected if not specified)"
    )
    parser.add_argument(
        "--corrupted_instructions_map",
        type=str,
        default="../../data/corrupted-data/corrupted_objects.json",
        help="Path to JSON mapping instructions to corrupted token indices",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--num_bins", type=int, default=1000)
    parser.add_argument("--torch_compile", action="store_true", help="Use torch.compile for model")
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="tracing_results.parquet")
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--save_interval", type=int, default=50, help="How often to write to disk")
    parser.add_argument("--trace_target", type=str, default="language", help="Target for causal tracing")

    args = parser.parse_args()
    if args.trace_target not in ["language", "action"]:
        raise ValueError("Invalid trace_target. Must be 'language' or 'action'.")
    if os.path.exists(args.output_path):
        raise FileExistsError(f"Output file {args.output_path} already exists. Please specify a different path or remove the existing file to avoid overwriting.")
    return args


@dataclass
class CorruptedDataArguments:
    repo_id: str = "HuggingFaceVLA/libero"
    history: int = 1
    horizon: int = 8
    img_size: int = 224
    crop_ratio: float = 0.875
    tile_images: bool = True
    brightness_aug: float = 0.2
    contrast_aug: float = 0.2
    saturation_aug: float = 0.2
    hue_aug: float = 0.05

def main():
    
    args= parse_args()
    data_args = CorruptedDataArguments()

    dataset = CorruptedLiberoDataset(
        repo_id=data_args.repo_id,
        history=data_args.history,
        horizon=data_args.horizon,
        img_size=data_args.img_size,
        crop_ratio=data_args.crop_ratio,
        tile_images=data_args.tile_images,
        brightness_aug=data_args.brightness_aug,
        contrast_aug=data_args.contrast_aug,
        saturation_aug=data_args.saturation_aug,
        hue_aug=data_args.hue_aug,
        corrupted_instructions_map=args.corrupted_instructions_map,
    )

    model = QwenVLActor(
        model_path=args.model_path,
        stats_path=args.stats_path,
        horizon=args.horizon,
        action_dim=args.action_dim,
        num_bins=args.num_bins,
        torch_compile=args.torch_compile,
    )


    tracer = CausalTracer(model)
    buffer = []
    total_to_process = len(dataset) // 8
    pbar = tqdm(range(total_to_process), desc="Tracing Progress")
    
    for i in pbar:
        try:
            idx = random.randint(0, len(dataset) - 1)
            data = dataset[idx]
            

            trace_out = tracer.trace_frame(
                data, 
                layer_range=range(24, 36), 
                noise_std=args.noise_std,
                trace_target=args.trace_target,
            )
            
            row = {
                "sample_idx": idx,
                "clean_token_entropies": trace_out["clean_token_entropies"],
                "corrupted_token_entropies": trace_out["corrupted_token_entropies"],
                "clean_max_probs": trace_out["clean_max_probs"],
                "corrupted_max_probs": trace_out["corrupted_max_probs"],
                "baseline_dist": trace_out["baseline_dist"],
                "instruction": trace_out["metadata"]["instruction"],
                "frame_idx": int(trace_out["metadata"]["frame_idx"]),
                "episode_idx": int(trace_out["metadata"]["episode_idx"]),
                "timestamp": float(trace_out["metadata"]["timestamp"])
            }

            for l_idx, val in enumerate(trace_out["recovery_curve"]):
                row[f"layer_{l_idx:02d}"] = val
            
            buffer.append(row)

            if (i + 1) % args.save_interval == 0:
                new_df = pd.DataFrame(buffer)
                
                if os.path.exists(args.output_path):
                    existing_df = pd.read_parquet(args.output_path, engine="pyarrow")
                    new_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                new_df.to_parquet(args.output_path, index=False, engine="pyarrow")
                buffer = [] # Clear RAM
                pbar.set_description(f"Checkpoint Saved: {args.output_path}")

        except Exception as e:
            print(f"\n[Error] Skipping index {idx}: {e}")
            continue

    if buffer:
        final_df = pd.DataFrame(buffer)
        if os.path.exists(args.output_path):
            existing_df = pd.read_parquet(args.output_path, engine="pyarrow")
            final_df = pd.concat([existing_df, final_df], ignore_index=True)
        final_df.to_parquet(args.output_path, index=False)

    print(f"\nExperiment Complete. Results stored at: {args.output_path}")


if __name__ == "__main__":
    main()