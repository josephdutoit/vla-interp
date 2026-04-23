# #!/usr/bin/env python3
# """VLA-0 Training Script using TRL's SFTTrainer."""
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

# import json
# from dataclasses import dataclass, field
# from pathlib import Path

# # from trl import SFTConfig, SFTTrainer, TrlParser

# from rv_train.collator import VLACollator
# from rv_train.dataset import LiberoDataset
# from rv_train.model import load_model_for_training, load_processor


# @dataclass
# class ModelArguments:
#     model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
#     use_flash_attention: bool = field(default=False)


# @dataclass
# class DataArguments:
#     repo_id: str = field(default="physical-intelligence/libero")
#     history: int = field(default=1)
#     horizon: int = field(default=8)
#     img_size: int = field(default=224)
#     crop_ratio: float = field(default=0.875)
#     tile_images: bool = field(default=True)
#     brightness_aug: float = field(default=0.2)
#     contrast_aug: float = field(default=0.2)
#     saturation_aug: float = field(default=0.2)
#     hue_aug: float = field(default=0.05)


# @dataclass
# class VLATrainingArguments:
#     action_mask_aug_pct: float = field(default=0.4)


# def main():
#     # parser = TrlParser(dataclass_types=[ModelArguments, DataArguments, VLATrainingArguments, SFTConfig])
#     # model_args, data_args, vla_args, training_args = parser.parse_args_and_config()

#     # print(f"Loading model: {model_args.model_id}")
#     # model = load_model_for_training(
#     #     model_id=model_args.model_id,
#     #     use_flash_attention=model_args.use_flash_attention,
#     # )

#     # processor = load_processor(
#     #     model_id=model_args.model_id,
#     #     img_size=data_args.img_size,
#     #     num_cams=2,
#     #     tile_images=data_args.tile_images,
#     # )

#     data_args = DataArguments()
#     print("Loading dataset...")
#     dataset = LiberoDataset(
#         repo_id=data_args.repo_id,
#         history=data_args.history,
#         horizon=data_args.horizon,
#         img_size=data_args.img_size,
#         crop_ratio=data_args.crop_ratio,
#         tile_images=data_args.tile_images,
#         brightness_aug=data_args.brightness_aug,
#         contrast_aug=data_args.contrast_aug,
#         saturation_aug=data_args.saturation_aug,
#         hue_aug=data_args.hue_aug,
#     )

#     # Save stats for inference
#     output_dir = "."
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     with open(f"{output_dir}/dataset_stats.json", "w") as f:
#         json.dump(dataset.stats, f, indent=2)

#     # collator = VLACollator(
#     #     processor=processor,
#     #     action_mask_aug_pct=vla_args.action_mask_aug_pct,
#     # )

#     # VLM-specific settings
#     # training_args.max_length = None  # Don't truncate images
#     # training_args.remove_unused_columns = False
#     # training_args.dataset_kwargs = {"skip_prepare_dataset": True}

#     # trainer = SFTTrainer(
#     #     model=model,
#     #     args=training_args,
#     #     train_dataset=dataset,
#     #     data_collator=collator,
#     #     processing_class=processor,
#     # )

#     print("Starting training...")
#     # trainer.train()

#     print("Saving final model...")
#     # trainer.save_model(f"{training_args.output_dir}/final")
#     # processor.save_pretrained(f"{training_args.output_dir}/final")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""VLA-0 Training Script: Fixed for LIBERO v3.0 Compatibility."""
import sys, os
import json
from dataclasses import dataclass, field
from pathlib import Path

# Ensure local src is in path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rv_train.dataset import LiberoDataset

@dataclass
class DataArguments:
    # Use the verified mirror to avoid the NotImplementedError
    repo_id: str = field(default="HuggingFaceVLA/libero")
    history: int = field(default=1)
    horizon: int = field(default=8)
    img_size: int = field(default=224)
    crop_ratio: float = field(default=0.875)
    tile_images: bool = field(default=True)
    # Augmentations
    brightness_aug: float = field(default=0.2)
    contrast_aug: float = field(default=0.2)
    saturation_aug: float = field(default=0.2)
    hue_aug: float = field(default=0.05)

def main():
    data_args = DataArguments()
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Environment Check ---")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Loading dataset: {data_args.repo_id}")

    try:
        # Initializing our wrapper which calls LeRobotDataset internally
        dataset = LiberoDataset(
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
        )

        print("--- Dataset Loaded Successfully ---")
        
        # LeRobot v3.0 stores stats in dataset.meta.stats
        # We extract and save them for the inference actor
        # stats = dataset.stats
        # stats_path = output_dir / "dataset_stats.json"
        
        # with open(stats_path, "w") as f:
        #     json.dump(stats, f, indent=2)
        
        # print(f"Successfully saved stats to: {stats_path}")
        # print(f"Action stats: {stats.get('action', 'Not Found')}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        if "NotImplementedError" in str(e):
            print("\nHINT: Your cache might still be poisoned with old metadata.")
            print("Run: rm -rf ~/.cache/huggingface/lerobot/physical-intelligence/libero")
        raise e

    print("\nInitialization Complete. Model training logic can be un-commented below.")
    # TODO: Initialize model, processor, and SFTTrainer here.

if __name__ == "__main__":
    main()