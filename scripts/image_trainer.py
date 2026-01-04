#!/usr/bin/env python3
"""
image-site
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time
import yaml
import toml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType



def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    """Merge default config with model-specific overrides."""
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None
def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    """Get configuration overrides based on model name."""
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    """Load the appropriate LRS configuration based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def load_size_config(model_type: str, is_style: bool) -> dict:
    """Load size-based configuration (size_person.json or size_style.json)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")
    
    if model_type == "flux":
        # Flux doesn't use size-based configs yet
        return {}
    
    config_file = os.path.join(config_dir, "size_style.json" if is_style else "size_person.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load size config from {config_file}: {e}", flush=True)
        return {}


def get_config_for_dataset_size(size_config: dict, num_images: int) -> dict:
    """Find and return config for the appropriate size range based on image count"""
    if not size_config or "size_ranges" not in size_config:
        return {}
    
    size_ranges = size_config.get("size_ranges", [])
    
    for range_config in size_ranges:
        min_size = range_config.get("min", 0)
        max_size = range_config.get("max", float('inf'))
        
        if min_size <= num_images <= max_size:
            config = range_config.get("config", {})
            print(f"Found size-based config for {num_images} images (range {min_size}-{max_size}): {config}", flush=True)
            return config
    
    print(f"Warning: No size range found for {num_images} images, using defaults", flush=True)
    return {}


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        # Load base config template
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        # Count images first (needed for size-based config)
        num_images = 0
        if os.path.exists(train_data_dir):
            for root, dirs, files in os.walk(train_data_dir):
                num_images += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        
        print(f"Detected {num_images} images for task {task_id}", flush=True)

        # Step 1: Load and apply size-based configuration (base layer)
        size_config = load_size_config(model_type, is_style)
        size_based_settings = get_config_for_dataset_size(size_config, num_images)
        
        if size_based_settings:
            print(f"Applying size-based configuration for {num_images} images", flush=True)
            
            # Apply size-based parameters
            for size_key in [
                "network_dim",
                "network_alpha", 
                "train_batch_size",
                "lr_scheduler",
                "lr_warmup_steps",
                "learning_rate",
                "unet_lr",
                "text_encoder_lr",
            ]:
                if size_key in size_based_settings:
                    config[size_key] = size_based_settings[size_key]
                    print(f"  {size_key}: {size_based_settings[size_key]}", flush=True)
        
        # Step 2: Load and apply model-specific overrides (top layer)
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                print(f"Applying model-specific overrides for '{model_name}'", flush=True)
                for optional_key in [
                    "max_grad_norm",
                    "prior_loss_weight",
                    "max_train_epochs",
                    "train_batch_size",
                    "optimizer_args",
                    "unet_lr",
                    "text_encoder_lr",
                    "noise_offset",
                    "min_snr_gamma",
                    "seed",
                    "lr_warmup_steps",
                    "lr_scheduler",
                    "loss_type",
                    "huber_c",
                    "huber_schedule",
                    "network_dim",
                    "network_alpha",
                ]:
                    if optional_key in lrs_settings:
                        config[optional_key] = lrs_settings[optional_key]
                        print(f"  override {optional_key}: {lrs_settings[optional_key]}", flush=True)
            else:
                print(f"No model-specific configuration found for '{model_name}'", flush=True)
        
        # Step 3: Apply network_args with smart dropout (always apply)
        if model_type == "sdxl":
            # Calculate base conv dims based on network_dim (if not already set by size config)
            network_dim = config.get("network_dim", 128)
            
            # Default conv dims scale with network_dim
            if network_dim <= 64:
                conv_dim, conv_alpha = 8, 4
            elif network_dim <= 128:
                conv_dim, conv_alpha = 16, 8
            else:
                conv_dim, conv_alpha = 32, 16
            
            # Smart dropout based on dataset size and task type
            if not is_style:
                dropout = 0.005  # Person tasks: minimal dropout
            elif num_images <= 12:
                dropout = 0.01   # Small style datasets: moderate dropout
            else:
                dropout = 0.1    # Larger style datasets: standard dropout
            
            config["network_args"] = [
                f"conv_dim={conv_dim}",
                f"conv_alpha={conv_alpha}",
                f"dropout={dropout}"
            ]
            print(f"  network_args: conv_dim={conv_dim}, conv_alpha={conv_alpha}, dropout={dropout}", flush=True)

        # Set basic paths
        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-script/{model_type}_train_network.py",
            "--config_file", config_path
        ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path
            ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word,
    )

    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
