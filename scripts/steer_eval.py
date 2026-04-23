#!/usr/bin/env python3
"""
Steering evaluation script for VLA-0 using SAE features.
Intervenes on model activations by scaling specific SAE features.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import argparse
from datetime import datetime
from pathlib import Path
import torch

from rv_interp.models.sae_vla import SAEQwenVLActor
from rv_interp.analysis.steering_evaluator import SteeringEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate steered VLA-0 on LIBERO")

    # Model & SAE
    parser.add_argument("--model_path", type=str, default="../logs/runs/vla0-trl-test", help="Path to trained model checkpoint")
    parser.add_argument("--sae_path", type=str, default="../../data/sae_checkpoints/sae_l11_e6.pt", help="Path to SAE checkpoint")
    parser.add_argument("--layer_idx", type=int, default=11, help="Layer where SAE was trained")
    parser.add_argument(
        "--stats_path", type=str, default="../logs/runs/vla0-trl-test/dataset_stats.json", help="Path to dataset_stats.json (auto-detected if not specified)"
    )

    # Steering settings
    parser.add_argument(
        "--steering", 
        type=str, 
        default=None, 
        help="Feature steering config. Format: 'idx1:weight1,idx2:weight2'. Weight > 1 amplifies, < 1 dampens."
    )

    # Evaluation settings
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_object",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="Task suite to evaluate",
    )
    parser.add_argument("--task_name", type=str, default=None, help="Specific task to evaluate")
    parser.add_argument("--num_episodes", type=int, default=-1, help="Number of tasks to evaluate")

    # Hyperparameters
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--frame_skip", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ensemble_prediction", type=int, default=1)
    parser.add_argument("--torch_compile", action="store_true", help="Use torch.compile for model")

    # Image processing
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--crop_ratio", type=float, default=0.875)
    parser.add_argument("--tile_images", action="store_true", default=True)

    # Sharding
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)

    # Output
    parser.add_argument("--log_dir", type=str, default="data/steering_outputs")
    parser.add_argument("--no_video", dest="save_video", action="store_false", default=True)
    parser.add_argument("--save_path", type=str, default="../../data/steering_outputs", help="Path to save evaluation directory")
    return parser.parse_args()


def parse_steering_weights(steering_str, d_sae, device):
    """Parses 'idx1:weight1,idx2:weight2' into a (d_sae,) tensor."""
    if not steering_str:
        return None
    
    weights = torch.ones(d_sae, device=device)
    for part in steering_str.split(","):
        idx_str, weight_str = part.split(":")
        weights[int(idx_str)] = float(weight_str)
    
    return weights


def build_log_dir(args, timestamp: str) -> str:
    """Build log directory path: logs/eval_logs/steered/{steering_name}/{timestamp}"""
    steering_name = args.steering.replace(":", "_").replace(",", "__") if args.steering else "no_steering"
    return str(Path("eval_logs") / "steered" / steering_name / timestamp)


def main():
    args = parse_args()

    # Auto-detect stats path
    stats_path = args.stats_path
    if stats_path is None:
        model_dir = Path(args.model_path).parent
        candidate = model_dir / "dataset_stats.json"
        if candidate.exists():
            stats_path = str(candidate)
        else:
            candidate = model_dir.parent / "dataset_stats.json"
            if candidate.exists():
                stats_path = str(candidate)

    if stats_path is None:
        raise ValueError("Could not find dataset_stats.json. Specify --stats_path")

    # Build log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or build_log_dir(args, timestamp)

    print(f"Loading model from: {args.model_path}")
    print(f"Loading SAE from: {args.sae_path}")
    print(f"Steering: {args.steering or 'None'}")
    print(f"Logs will be saved to: {log_dir}")

    # Initialize SAE Actor
    model = SAEQwenVLActor(
        model_path=args.model_path,
        sae_path=args.sae_path,
        layer_idx=args.layer_idx,
        stats_path=stats_path,
        horizon=8,  # Default for QwenVLActor
        action_dim=7,
        num_bins=1000,
        torch_compile=args.torch_compile,
    )

    # Set steering weights
    steering_weights = parse_steering_weights(args.steering, model.d_sae, model.device)
    model.set_steering_weights(steering_weights)

    # Initialize Evaluator
    evaluator = SteeringEvaluator(
        model=model,
        log_dir=log_dir,
        save_video=args.save_video,
        seed=args.seed,
        action_horizon=args.action_horizon,
        frame_skip=args.frame_skip,
        img_size=args.img_size,
        crop_ratio=args.crop_ratio,
        tile_images=args.tile_images,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        ensemble_prediction=args.ensemble_prediction,
    )

    print(f"Starting steered evaluation on {args.task_suite}, {args.task_name}...")

    evaluator.setup(args.task_suite, args.task_name)

    evaluator.evaluate(
        save_path=os.path.join(
            args.save_path, 
            f"{args.steering.replace(':', '_').replace(',', '__') if args.steering else 'no_steering'}_{args.task_suite}_{args.task_name}.mp4"
        )
    )

    print(f"\nResults saved to: {log_dir}/")


if __name__ == "__main__":
    main()
