from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "sshleifer/tiny-gpt2"
MODEL_DIR = PROJECT_ROOT / "models" / "tiny-gpt2"

DEFAULT_SEQ_LEN = 128
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100
