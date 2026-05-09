from pathlib import Path
from uuid import uuid4


DATA_DIR = Path("data")
TASK_DIR = DATA_DIR / "tasks"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"


def ensure_data_dirs():
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def create_task_workdir(task_id: str | None = None) -> Path:
    ensure_data_dirs()

    if task_id is None:
        task_id = uuid4().hex

    work_dir = TASK_DIR / task_id
    (work_dir / "input").mkdir(parents=True, exist_ok=True)
    (work_dir / "output").mkdir(parents=True, exist_ok=True)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)

    return work_dir


def save_upload_file(file_bytes: bytes, filename: str) -> Path:
    ensure_data_dirs()

    safe_name = filename.replace("/", "_").replace("\\", "_")
    upload_path = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"

    with upload_path.open("wb") as f:
        f.write(file_bytes)

    return upload_path