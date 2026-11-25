import copy
import tomllib
from pathlib import Path

DEFAULTS = {
    "audio": {
        "source": "loopback",
        "device": "",
        "file": "",
        "file_realtime": True,
    },
    "asr": {
        "model": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
        "language": "zh",
        "beam_size": 1,
        "download_root": "models",
        "initial_prompt": "以下是普通话直播的内容，使用简体中文和标点符号。",
        "to_simplified": True,
        "hotwords": "",
        "corrections": "",
    },
    "segment": {
        "step_sec": 0.4,
        "silence_commit_sec": 0.6,
        "min_chars": 6,
        "soft_max_chars": 18,
        "max_chars": 30,
        "max_sec": 9.0,
        "pause_gap_sec": 0.25,
        "clause_wait_sec": 0.7,
        "vad_threshold": 0.5,
    },
    "translate": {
        "model": "gpt-6-luna",
        "effort": "none",
        "context_segments": 4,
        "draft": False,
        "draft_interval_sec": 1.2,
        "draft_min_chars": 8,
        "draft_model": "",
        "style": "",
        "glossary": "",
        "glossary_file": "",
        "sound_alike": True,
        "refine": True,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
        "text_file": "",
        "text_file_lines": 2,
        # OBS 에서 방송 화면·소리를 늦춘 초. 오버레이가 자막을 말 시작 + delay 에 띄우고, 그 사이 다음 말을 참고해 다시 번역
        "delay": 0,
    },
}


def _merge(base: dict, over: dict) -> dict:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    return cfg
