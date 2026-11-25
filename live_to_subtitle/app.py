"""전체 파이프라인: 오디오 -> 문장 분할(인식) -> 번역 -> OBS 오버레이(WebSocket)."""
import asyncio
import json
import queue
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aiohttp import WSMsgType, web

from .asr import ASR
from .audio import FileSource, create_source
from .segmenter import Segmenter, bare_text, char_len
from .translator import Translator

STATIC = Path(__file__).parent / "static"
KEEP_SEGMENTS = 30
# 확정 시점에 이미 진행 중인 (같은 문장의) 번역이 이 시간 안에 안 끝나면 새 요청을 하나 더 보냄
HEDGE_SEC = 1.5
# 다시 번역(refine)을 자막 표시 마감보다 이만큼 먼저 시작 (번역 요청 p90 ≈ 1.6초)
REFINE_LEAD_SEC = 1.8


def _ignore_disconnect_noise(loop: asyncio.AbstractEventLoop, context: dict):
    """Windows 에서 브라우저(OBS)가 연결을 끊을 때 asyncio 가 찍는 무해한 WinError 10022/10054 트레이스백 숨김."""
    exc = context.get("exception")
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) in (10022, 10054):
        return
    loop.default_exception_handler(context)


@dataclass
class Segment:
    id: int
    zh: str = ""
    ko: str = ""
    # live: 말하는 중 / final: 중국어 확정 / done: 번역 확정 / skip: 번역할 내용 없음
    state: str = "live"
    ko_draft: bool = False
    # 오디오상 말 시작/끝 (입력 시작 기준 초)
    t_start: float = 0.0
    t_end: float = 0.0
    # 말 시작을 들은 실제 시각 (epoch 초). 오버레이 delay 옵션이 이 시각 + delay 에 자막을 띄움
    spoken_at: float = 0.0
    updated: float = field(default_factory=time.time)


@dataclass
class TranslateJob:
    """세그먼트 하나에 대해 진행 중이거나 끝난 번역 요청."""

    zh: str
    # True 면 "아직 말하는 중, 문장이 잘렸을 수 있음" 이라고 모델에 알림 (말하는 도중의 임시 번역)
    cut_off: bool
    started: float = field(default_factory=time.monotonic)
    task: asyncio.Task | None = None
    ko: str | None = None  # 끝난 번역 ("" = 건너뜀, None = 아직 / 실패)


class App:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # 실행마다 다른 값. 세그먼트 번호가 1부터 다시 시작하므로, 열려 있던 OBS 오버레이가
        # 재접속했을 때 이전 실행의 자막을 비우게 함
        self.session = f"{time.time():.3f}"
        self.loop: asyncio.AbstractEventLoop | None = None
        self.segments: dict[int, Segment] = {}
        self.clients: set[web.WebSocketResponse] = set()
        self.translator = Translator(cfg["translate"])
        # 세그먼트별 최근 번역 요청. 확정된 문장이 이 요청의 문장과 같으면 다시 번역하지 않고 결과를 그대로 씀
        self.jobs: dict[int, TranslateJob] = {}
        self.background: set[asyncio.Task] = set()
        # 화면(TUI) 표시용 상태
        self.status = "시작 중"
        self.paused = False  # True 면 새로 확정되는 문장을 번역·표시하지 않음 (광고·쉬는 시간 등)
        self.segmenter: Segmenter | None = None
        self.base_url = ""
        # 세그먼트별 시각: end = 말이 끝난 시각, final = 최종 번역이 나온 시각 (epoch), refined = 다시 번역 반영 여부
        self.timing: dict[int, dict] = {}
        self.refine_tried = 0
        self.refine_applied = 0
        self.key_status = ""  # 번역 API 연결 확인 결과 (Translator.warmup)

    # ---------------------------------------------------------- segments
    def _seg(self, seg_id: int) -> Segment:
        seg = self.segments.get(seg_id)
        if seg is None:
            seg = self.segments[seg_id] = Segment(seg_id)
            for old in sorted(self.segments)[:-KEEP_SEGMENTS]:
                del self.segments[old]
        return seg

    async def _publish(self, seg: Segment):
        seg.updated = time.time()
        msg = json.dumps({"t": "seg", **asdict(seg)}, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_str(msg)
            except (ConnectionError, RuntimeError):
                self.clients.discard(ws)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        return task

    # 세그먼터 스레드에서 호출됨
    def on_partial(self, seg_id: int, text: str, paused: bool, spoken_at: float = 0.0):
        self.loop.call_soon_threadsafe(self._on_partial, seg_id, text, paused, spoken_at)

    def on_commit(self, seg_id: int, text: str, t_start: float, t_end: float, spoken_at: float = 0.0):
        self.loop.call_soon_threadsafe(self._on_commit, seg_id, text, t_start, t_end, spoken_at)

    def _on_partial(self, seg_id: int, text: str, paused: bool, spoken_at: float = 0.0):
        if self.paused:
            return
        seg = self._seg(seg_id)
        if seg.state != "live":
            return
        seg.zh = text
        if spoken_at:
            seg.spoken_at = spoken_at
        self._spawn(self._publish(seg))
        if not text:
            return

        job = self.jobs.get(seg_id)
        busy = job is not None and not job.task.done()
        if paused:
            # 말이 멈춤 -> 이 문장이 곧 그대로 확정될 가능성이 큼. 확정을 기다리지 않고 바로 번역 시작
            # (확정된 문장이 같으면 이 결과가 그대로 최종 번역이 됨)
            if job and bare_text(job.zh) == bare_text(text) and (busy or job.ko):
                return
            self._start_job(seg, text, cut_off=False)
            return

        tcfg = self.cfg["translate"]
        if not tcfg["draft"] or char_len(text) < tcfg["draft_min_chars"] or busy:
            return
        if job and (time.monotonic() - job.started < tcfg["draft_interval_sec"] or job.zh == text):
            return
        self._start_job(seg, text, cut_off=True)

    def _on_commit(self, seg_id: int, text: str, t_start: float, t_end: float, spoken_at: float = 0.0):
        seg = self._seg(seg_id)
        seg.zh = text
        if self.paused:
            seg.state = "skip"
            self._spawn(self._publish(seg))
            return
        seg.state = "final"
        seg.t_start, seg.t_end = t_start, t_end
        if spoken_at:
            seg.spoken_at = spoken_at
        print(f"[zh #{seg_id}] {text}")
        self._spawn(self._publish(seg))

        job = self.jobs.get(seg_id)
        if job and bare_text(job.zh) == bare_text(text):
            if not job.task.done():
                # 같은 문장을 번역 중: 끝나면 그대로 최종 번역이 됨.
                # 그 요청이 서버에서 늦어지면 새 요청을 하나 더 보내 먼저 끝나는 쪽을 씀
                self._spawn(self._hedge(seg, job))
                return
            if job.ko:
                self._spawn(self._finish(seg, job.ko))
                return
        self._start_job(seg, text, cut_off=False)

    def _start_job(self, seg: Segment, zh: str, cut_off: bool, keep_old: bool = False):
        old = self.jobs.get(seg.id)
        if old and not old.task.done() and not keep_old:
            old.task.cancel()
        job = self.jobs[seg.id] = TranslateJob(zh, cut_off)
        job.task = self._spawn(self._translate(seg, job))

    async def _hedge(self, seg: Segment, job: TranslateJob):
        await asyncio.wait([job.task], timeout=HEDGE_SEC)
        if not job.task.done() and seg.state == "final" and self.jobs.get(seg.id) is job:
            self._start_job(seg, seg.zh, cut_off=False, keep_old=True)

    async def _translate(self, seg: Segment, job: TranslateJob):
        # 화면에 이미 번역이 떠 있으면 새 번역이 다 끝난 뒤 한 번에 교체 (깜빡임 방지),
        # 아무것도 없으면 토큰 단위로 바로 흘려보냄 (지연 최소화)
        stream_live = not seg.ko
        show_draft = self.cfg["translate"]["draft"]

        def current() -> bool:
            return self.jobs.get(seg.id) is job

        async def on_text(partial: str):
            if stream_live and current() and (show_draft or seg.state == "final"):
                seg.ko, seg.ko_draft = partial, True
                await self._publish(seg)

        ko = job.ko = await self.translator.translate(seg.id, job.zh, job.cut_off, on_text)
        if seg.state == "live":  # 아직 말하는 중: 임시 번역으로 표시
            if ko and show_draft and current():
                seg.ko, seg.ko_draft = ko, True
                await self._publish(seg)
            return
        # 확정된 문장과 같은 문장의 번역이면 (승격된 요청 / 추가 요청 중) 먼저 끝난 쪽을 씀
        if seg.state == "final" and bare_text(job.zh) == bare_text(seg.zh):
            if ko is None and not current():
                return  # 실패했지만 다른 요청이 아직 진행 중
            await self._finish(seg, ko)

    async def _finish(self, seg: Segment, ko: str | None):
        """확정된 문장(final)에 최종 번역을 반영."""
        if seg.state != "final":
            return
        job = self.jobs.pop(seg.id, None)
        if job and not job.task.done() and job.task is not asyncio.current_task():
            job.task.cancel()  # 함께 보낸 다른 요청은 더 필요 없음
        if ko is None:  # 실패: 초안이 있으면 그대로 두고 확정 표시만
            seg.state = "done"
            seg.ko_draft = False
            await self._publish(seg)
            return
        seg.ko, seg.ko_draft = ko, False
        seg.state = "done" if ko else "skip"
        print(f"[ko #{seg.id}] {ko or '(skip)'}")
        if ko and seg.spoken_at:
            self.timing[seg.id] = {"start": seg.spoken_at, "end": seg.spoken_at + seg.t_end - seg.t_start,
                                   "final": time.time(), "refined": False}
            for old in sorted(self.timing)[:-500]:
                del self.timing[old]
        self.translator.remember(seg.id, seg.zh, ko)
        await self._publish(seg)
        self._write_text_file()
        if ko and self.cfg["translate"].get("refine", True) and self.cfg["server"].get("delay", 0) > 0:
            self._spawn(self._refine(seg))

    async def _refine(self, seg: Segment):
        """방송을 늦춘 경우: 자막이 화면에 뜨기 직전에, 그 사이 들린 다음 말을 참고해 다시 번역.
        중간에 잘린 문장이나 잘못 인식된 말을 뒷말로 바로잡는다 (오버레이는 화면에 뜬 자막은 바꾸지 않음)."""
        if not seg.spoken_at:
            return
        due = seg.spoken_at + self.cfg["server"]["delay"]
        wait = due - REFINE_LEAD_SEC - time.time()
        if wait < 0.2:
            return  # 이미 마감이 가까움: 첫 번역을 그대로 씀
        await asyncio.sleep(wait)
        following = self._following(seg.id)
        if char_len(following) < 4 or seg.state != "done":
            return
        zh, base = seg.zh, seg.ko
        self.refine_tried += 1
        ko = await self.translator.translate(seg.id, zh, False, following=following)
        if not ko or ko == base or seg.zh != zh or seg.ko != base or time.time() > due - 0.05:
            return  # 실패·같음·마감 지남 -> 첫 번역 유지
        seg.ko = ko
        self.refine_applied += 1
        if seg.id in self.timing:
            self.timing[seg.id]["refined"] = True
        print(f"[ko+ #{seg.id}] {ko}")
        self.translator.remember(seg.id, zh, ko)
        await self._publish(seg)
        self._write_text_file()

    def _following(self, seg_id: int) -> str:
        """이 줄 다음에 지금까지 들린 말 (확정된 줄 + 인식 중인 줄)."""
        parts = [s.zh for i, s in sorted(self.segments.items()) if i > seg_id and s.zh and s.state != "skip"]
        return "".join(parts[:3])[:80]

    def _write_text_file(self):
        path = self.cfg["server"]["text_file"]
        if not path:
            return
        done = [s.ko for _, s in sorted(self.segments.items()) if s.state == "done" and s.ko]
        lines = done[-self.cfg["server"]["text_file_lines"] :]
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(path)

    def _hello(self) -> str:
        return json.dumps({"t": "hello", "session": self.session, "now": time.time(),
                           "delay": self.cfg["server"].get("delay", 0)})

    async def set_delay(self, sec: float):
        """방송 지연을 바꿈 (오버레이에도 바로 알림). OBS 의 지연 필터는 따로 맞춰야 함."""
        self.cfg["server"]["delay"] = max(0.0, sec)
        msg = self._hello()
        for ws in list(self.clients):
            try:
                await ws.send_str(msg)
            except (ConnectionError, RuntimeError):
                self.clients.discard(ws)

    # ---------------------------------------------------------- web
    async def ws_handler(self, request: web.Request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(self._hello())
        for _, seg in sorted(self.segments.items()):
            await ws.send_str(json.dumps({"t": "seg", **asdict(seg)}, ensure_ascii=False))
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
        self.clients.discard(ws)
        return ws

    def make_web_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", lambda r: web.FileResponse(STATIC / "overlay.html"))
        app.router.add_get("/log", lambda r: web.FileResponse(STATIC / "log.html"))
        app.router.add_get("/preview", lambda r: web.FileResponse(STATIC / "preview.html"))
        app.router.add_get("/ws", self.ws_handler)
        return app

    # ---------------------------------------------------------- main
    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.loop.set_exception_handler(_ignore_disconnect_noise)
        scfg = self.cfg["server"]

        runner = web.AppRunner(self.make_web_app())
        await runner.setup()
        await web.TCPSite(runner, scfg["host"], scfg["port"]).start()
        host = "127.0.0.1" if scfg["host"] in ("0.0.0.0", "") else scfg["host"]
        base = self.base_url = f"http://{host}:{scfg['port']}"
        print(f"[server] OBS 브라우저 소스: {base}/   (모니터링: {base}/log)")
        self.status = "모델 불러오는 중"

        asr, self.key_status = await asyncio.gather(asyncio.to_thread(ASR, self.cfg["asr"]), self.translator.warmup())
        audio_q: queue.Queue = queue.Queue()
        source = self.source = create_source(self.cfg["audio"], audio_q)
        segmenter = Segmenter(audio_q, asr, self.cfg["segment"], self.cfg["asr"], self.on_partial, self.on_commit)
        self.segmenter = segmenter
        segmenter.start()
        source.start()
        self.status = "듣는 중"
        print("[ready] 듣는 중... (Ctrl+C 로 종료)")

        try:
            if isinstance(source, FileSource):
                await asyncio.to_thread(source.finished.wait)
                await asyncio.sleep(1.5)  # 남은 오디오 처리
                segmenter.stop()
                await asyncio.to_thread(segmenter.join)
                segmenter.flush()
                await asyncio.sleep(0.1)  # flush 가 예약한 콜백 실행
                while self.background:
                    await asyncio.gather(*list(self.background), return_exceptions=True)
                    await asyncio.sleep(0)  # 끝난 작업을 background 에서 빼는 콜백이 돌 기회를 줌
                self.status = "파일 처리 완료"
                print("[done] 파일 처리 완료 (Ctrl+C 로 서버 종료)")
            await asyncio.Event().wait()
        finally:
            source.stop()
            segmenter.stop()
            await runner.cleanup()
