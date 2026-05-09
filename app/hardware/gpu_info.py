import subprocess
from typing import Any


def get_gpu_info() -> list[dict[str, Any]]:
    """
    NVIDIA GPU 정보를 가져온다.

    nvidia-smi가 없거나 GPU가 없으면 빈 리스트를 반환한다.
    """

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError:
        return []

    gpus = []

    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]

        if len(parts) != 7:
            continue

        index, name, total, used, free, temp, util = parts

        gpus.append(
            {
                "index": int(index),
                "name": name,
                "vram_total_mb": int(total),
                "vram_used_mb": int(used),
                "vram_free_mb": int(free),
                "temperature": int(temp),
                "utilization": int(util),
            }
        )

    return gpus