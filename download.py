import os
import logging
from huggingface_hub import snapshot_download
import huggingface_hub.utils.logging as hf_logging

# 1. Setup standard Python logging for your server
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# 2. THE FIX: Force the Hugging Face Hub library to spill all its secrets
hf_logging.set_verbosity_debug()


def download_model_files():
    repo_id = "Lightricks/LTX-2.3"
    local_dir = os.path.expanduser("~/sd_models/LTX-2-Wrapper")

    logging.info(f"Initiating connection to Hugging Face Hub...")
    logging.info(f"Target Repo: {repo_id}")
    logging.info(f"Destination: {local_dir}")

    try:
        # snapshot_download is the Python equivalent of 'hf download'
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            # allow_patterns Replaces the --include flag
            allow_patterns=["*.json", "tokenizer/*", "scheduler/*", "vae/*"],
            # ignore_patterns Replaces the --exclude flag
            ignore_patterns=[
                "*.bin",
                "*.pt",
                "*.ckpt",
                "transformer/*.safetensors",
                "text_encoder/*.safetensors",
            ],
            max_workers=4,  # Parallel downloads
            resume_download=True,  # Will safely pick up where it left off if it crashes
        )
        logging.info("✅ Download completed successfully!")

    except Exception as e:
        logging.error(f"❌ Download failed: {e}", exc_info=True)


if __name__ == "__main__":
    download_model_files()
