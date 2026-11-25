"""App 번역 작업 관리 검증 (가짜 번역기, GPU·API 키 불필요).

실행: .venv\\Scripts\\python.exe tests\\test_app.py
- 말 멈춤 번역의 승격 / 진행 중 요청의 승격 / 멈춘 요청 헤지 / 다른 문장 확정 / 임시 번역 승격
- 방송 지연 중 다음 말을 참고한 다시 번역(refine) 반영, 마감이 지난 다시 번역은 버림
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live_to_subtitle.app as appmod  # noqa: E402
from live_to_subtitle.config import load_config  # noqa: E402

appmod.HEDGE_SEC = 0.3
T = "我们今天来玩这个游戏"


class FakeTranslator:
    def __init__(self):
        self.calls = []
        self.delays = []  # 호출 순서대로 걸릴 시간

    async def translate(self, seg_id, zh, cut_off, on_text=None, following=""):
        d = self.delays.pop(0) if self.delays else 0.2
        self.calls.append((zh, cut_off, following))
        await asyncio.sleep(d)
        return f"KO({zh}|{following})" if following else f"KO({zh})"

    def remember(self, *a):
        pass


def make_app(**translate_cfg) -> appmod.App:
    cfg = load_config(None)
    cfg["translate"].update(draft=True, **translate_cfg)
    app = appmod.App(cfg)
    app.loop = asyncio.get_running_loop()
    app.translator = FakeTranslator()
    return app


async def drain(app):
    while app.background:
        await asyncio.gather(*list(app.background), return_exceptions=True)
        await asyncio.sleep(0)


async def scenario(name, steps, expect_calls, expect_ko, max_finish) -> bool:
    app = make_app()
    tr = app.translator
    t_commit = None
    for kind, arg, wait in steps:
        if kind == "delays":
            tr.delays = arg
        elif kind == "partial":
            app._on_partial(1, arg[0], arg[1])
        elif kind == "commit":
            t_commit = time.monotonic()
            app._on_commit(1, arg, 0, 0)
        await asyncio.sleep(wait)
    t0 = time.monotonic()
    while app.segments[1].state == "final" and time.monotonic() - t0 < 5:
        await asyncio.sleep(0.01)
    seg = app.segments[1]
    took = time.monotonic() - t_commit
    ok = seg.state == "done" and seg.ko == expect_ko and len(tr.calls) == expect_calls and took <= max_finish
    print(f"{'OK  ' if ok else 'FAIL'} {name}: state={seg.state} ko={seg.ko} calls={len(tr.calls)} 확정→최종 {took:.2f}s")
    await drain(app)
    return ok


async def refine(name, delay, expect_refined) -> bool:
    """방송 지연 delay 초. 말 시작 직후 확정 -> 번역 -> 마감 1.8초 전에 다음 말을 참고해 다시 번역."""
    app = make_app(refine=True)
    app.cfg["server"]["delay"] = delay
    now = time.time()
    app._on_commit(1, "其实我们这个", 0, 1.0, spoken_at=now)
    app._on_partial(2, "版本的重点是新地图", False, spoken_at=now + 1.2)
    await asyncio.sleep(max(0.0, delay - 1.0))
    await drain(app)
    seg = app.segments[1]
    refined = "|" in seg.ko
    ok = refined == expect_refined and seg.state == "done"
    print(f"{'OK  ' if ok else 'FAIL'} {name}: ko={seg.ko} 다시 번역 {app.refine_applied}/{app.refine_tried}")
    return ok


async def main() -> int:
    res = [
        # 1) 말 멈춤 번역이 확정 전에 끝남 -> 확정 즉시 최종
        await scenario("멈춤 번역 완료 후 확정", [("partial", (T, True), 0.3), ("commit", T + "。", 0)], 1, f"KO({T})", 0.05),
        # 2) 확정 시점에 진행 중 -> 그 결과 사용, 추가 요청 없음
        await scenario("진행 중 승격", [("delays", [0.25], 0), ("partial", (T, True), 0.1), ("commit", T, 0)], 1, f"KO({T})", 0.2),
        # 3) 진행 중 요청이 멈춤 -> 헤지 요청이 먼저 끝남
        await scenario("멈춘 요청 헤지", [("delays", [3.0, 0.1], 0), ("partial", (T, True), 0.1), ("commit", T, 0)], 2,
                       f"KO({T})", 0.5),
        # 4) 다른 문장으로 확정 -> 취소하고 새 요청
        await scenario("다른 문장 확정", [("partial", (T, True), 0.05), ("commit", T[:5], 0)], 2, f"KO({T[:5]})", 0.3),
        # 5) 일반 임시 번역(말하는 중) -> 같은 문장 확정이면 승격
        await scenario("임시 번역 승격", [("partial", (T, False), 0.3), ("commit", T, 0)], 1, f"KO({T})", 0.05),
        # 6) 지연 3초: 마감 1.8초 전에 다음 말을 참고해 다시 번역 -> 반영
        await refine("다시 번역 반영", 3.0, True),
        # 7) 지연 1.5초: 마감까지 시간이 없어 다시 번역하지 않음
        await refine("시간 없으면 다시 번역 안 함", 1.5, False),
    ]
    print("ALL OK" if all(res) else "SOME FAILED")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
