import argparse
import asyncio
import os
import sys
import time
import warnings
from pathlib import Path

from .config import load_config

# huggingface_hub 의 Windows 심볼릭 링크 경고 숨김
warnings.filterwarnings("ignore", message=".*symlinks.*")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def check(cfg: dict) -> int:
    """모델 다운로드 + GPU 동작 + API 키 점검."""
    import numpy as np

    from .asr import ASR

    asr = ASR(cfg["asr"])
    audio = np.zeros(16000 * 5, np.float32)
    asr.transcribe(audio, "")
    t = time.time()
    asr.transcribe(audio, "")
    print(f"[check] 음성 인식 OK ({cfg['asr']['device']}, 5초 오디오 {time.time() - t:.2f}s)")

    if os.environ.get("OPENAI_API_KEY"):
        print("[check] OPENAI_API_KEY 확인됨")
    else:
        print("[check] 참고: OPENAI_API_KEY 가 없습니다. 프로젝트 폴더의 .env 에 추가해야 번역이 동작합니다.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="live_to_subtitle", description="중국어 방송 -> 한국어 실시간 자막 (OBS)")
    parser.add_argument("--config", default="config.toml", help="설정 파일 (기본: config.toml)")
    parser.add_argument("--list-devices", action="store_true", help="오디오 장치 목록 출력")
    parser.add_argument("--check", action="store_true", help="모델 다운로드 및 GPU 점검")
    parser.add_argument("--file", help="테스트: 오디오/영상 파일을 입력으로 사용")
    parser.add_argument("--fast", action="store_true", help="--file 을 실시간이 아닌 최대 속도로 처리")
    parser.add_argument("--device", help="오디오 장치 이름 일부 (설정 덮어쓰기)")
    parser.add_argument("--input", action="store_true", help="loopback 대신 입력 장치(마이크 등) 사용")
    parser.add_argument("--port", type=int, help="오버레이 서버 포트")
    parser.add_argument("--plain", action="store_true", help="대시보드 화면 대신 로그만 출력")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    from dotenv import load_dotenv

    # .env 의 OPENAI_API_KEY (프로젝트 폴더 기준, 없으면 현재 폴더)
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path.cwd() / ".env")
    cfg = load_config(args.config)
    if args.file:
        cfg["audio"].update(source="file", file=args.file, file_realtime=not args.fast)
    if args.input:
        cfg["audio"]["source"] = "input"
    if args.device is not None:
        cfg["audio"]["device"] = args.device
    if args.port:
        cfg["server"]["port"] = args.port

    if args.list_devices:
        from .audio import list_devices

        list_devices()
        return 0
    if args.check:
        return check(cfg)

    from .app import App

    if not args.plain and sys.stdout.isatty():
        from .tui import run as run_tui

        run_tui(cfg, args.config)
        return 0
    try:
        asyncio.run(App(cfg).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
