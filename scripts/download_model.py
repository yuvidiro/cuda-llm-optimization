from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sshleifer/tiny-gpt2"
MODEL_DIR = ROOT / "models" / "tiny-gpt2"


def main() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID}")
    print(f"Destination: {MODEL_DIR}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
    )

    print("\nModel download complete.")
    print(f"Local model: {MODEL_DIR}")


if __name__ == "__main__":
    main()
