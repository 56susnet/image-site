#!/bin/bash

# Auto-cleanup old containers to prevent conflicts
docker rm -f image-trainer-example hf-uploader downloader-image 2>/dev/null || true

TASK_ID="1c93dd95-2e89-48d9-813d-e0f521599cfd"
MODEL="gradients-io-tournaments/Z-Image-Turbo"
DATASET_ZIP="https://s3.eu-central-003.backblazeb2.com/gradients-validator/6da713f69539e32b_train_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=00362e8d6b742200000000002%2F20251226%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20251226T182738Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=175387a86ac618003e39ee8d44558fef5bfa5b159839ad7d20bb3f0e6a5f0ac6"
MODEL_TYPE="z-image"
EXPECTED_REPO_NAME="test_zimage-1"

HUGGINGFACE_USERNAME=""
HUGGINGFACE_TOKEN=""
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 700 "$OUTPUTS_DIR"

echo "Downloading model and dataset..."
docker run --rm   --volume "$CHECKPOINTS_DIR:/cache:rw"   --name downloader-image   trainer-downloader   --task-id "$TASK_ID"   --model "$MODEL"   --dataset "$DATASET_ZIP"   --task-type "ImageTask"   --model-type "$MODEL_TYPE" 

echo "Starting image training..."
docker run --rm --gpus all   --security-opt=no-new-privileges   --cap-drop=ALL   --memory=32g   --cpus=8   --network none   --env TRANSFORMERS_CACHE=/cache/hf_cache   --volume "$CHECKPOINTS_DIR:/cache:rw"   --volume "$OUTPUTS_DIR:/app/checkpoints/:rw"   --name image-trainer-example   standalone-image-toolkit-trainer   --task-id "$TASK_ID"   --model "$MODEL"   --dataset-zip "$DATASET_ZIP"   --model-type "$MODEL_TYPE"   --expected-repo-name "$EXPECTED_REPO_NAME"   --hours-to-complete 1

echo "Uploading model to HuggingFace..."
docker run --rm --gpus all   --volume "$OUTPUTS_DIR:/app/checkpoints/:rw"   --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN"   --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME"   --env TASK_ID="$TASK_ID"   --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME"   --env LOCAL_FOLDER="$LOCAL_FOLDER"   --env HF_REPO_SUBFOLDER="checkpoints"   --name hf-uploader   hf-uploader
