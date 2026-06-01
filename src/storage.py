from pathlib import Path
import os
import shutil

from dotenv import load_dotenv


load_dotenv()


class LocalStorage:
    def __init__(self):
        self.raw_dir = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
        self.processed_dir = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))

    def list_raw_files(self, pattern: str = "*.csv"):
        return list(self.raw_dir.glob(pattern))

    def save_processed_file(self, source_file: Path, output_filename: str):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / output_filename
        shutil.copy(source_file, output_path)
        return output_path


def get_storage():
    backend = os.getenv("STORAGE_BACKEND", "local").lower()

    if backend == "local":
        return LocalStorage()

    raise ValueError(f"Unsupported storage backend: {backend}")
