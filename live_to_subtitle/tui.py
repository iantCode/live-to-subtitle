"""터미널 대시보드 (Textual). `run.bat` 의 기본 화면. `--plain` 이면 쓰지 않고 예전처럼 로그만 출력."""
import asyncio
import contextlib
import math
import os
import re
import statistics
import subprocess
import time
import webbrowser
from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual import events, work
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, RichLog, Static

from .app import App

# 자막 줄로 따로 보여 주므로 로그 창에는 넣지 않는 줄
_SUB_LINE = re.compile(r"^\[(zh|ko|ko\+) #\d+\]")
_BAD = re.compile(r"오류|실패|error|Error|Traceback|한도|경고|찾을 수 없")


def _meter(rms: float, width: int = 18) -> Text:
    """입력 음량 막대 (-60 ~ 0 dBFS)."""
    db = 20 * math.log10(rms) if rms > 1e-6 else -90.0
    fill = max(0, min(width, round((db + 60) / 60 * width)))
    color = "green" if db < -12 else ("yellow" if db < -3 else "red")
    t = Text("▮" * fill, style=color)
    t.append("▯" * (width - fill), style="grey35")
    t.append(f" {db:5.0f} dB" if db > -90 else "   무음", style="grey62")
    return t


def _save_delay(config_path: Path, sec: float) -> bool:
    """config.toml 의 [server] delay 값을 바꿔 저장 (주석·다른 설정은 그대로)."""
    if not config_path.exists():
        return False
    text = config_path.read_text(encoding="utf-8")
    m = re.search(r"(?ms)^\[server\]\s*$(.*?)(?=^\[|\Z)", text)
    if not m:
        return False
    body = m.group(1)
    value = f"{sec:g}"
    if re.search(r"(?m)^delay\s*=", body):
        body = re.sub(r"(?m)^delay\s*=\s*[^\n#]*", f"delay = {value}", body, count=1)
    else:
        body = body.rstrip("\n") + f"\ndelay = {value}\n\n"
    config_path.write_text(text[: m.start(1)] + body + text[m.end(1):], encoding="utf-8")
    return True


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _save_env(path: Path, name: str, value: str):
    """.env 에 NAME=value 를 저장 (같은 이름이 있으면 그 줄만 바꿈)."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        if re.match(rf"\s*{name}\s*=", line):
            if not done:
                out.append(f"{name}={value}")
                done = True
            continue
        out.append(line)
    if not done:
        out.append(f"{name}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _mask(key: str) -> str:
    return f"{key[:3]}…{key[-4:]}" if len(key) > 10 else ("설정됨" if key else "없음")


class ApiKeyScreen(ModalScreen):
    """OpenAI API 키 입력 (프로젝트 폴더의 .env 에 저장)."""

    DEFAULT_CSS = """
    ApiKeyScreen { align: center middle; }
    #dialog { width: 76; height: auto; border: round $accent; background: $surface; padding: 1 2; }
    #dialog Label { margin-bottom: 1; }
    #buttons { height: auto; align: right middle; margin-top: 1; }
    #buttons Button { margin-left: 2; }
    """
    BINDINGS = [Binding("escape", "cancel", "취소")]

    def compose(self) -> ComposeResult:
        current = os.environ.get("OPENAI_API_KEY", "")
        with Vertical(id="dialog"):
            yield Label("[b]OpenAI API 키[/b]  (현재: " + _mask(current) + ")")
            yield Label("https://platform.openai.com/api-keys 에서 만든 키를 붙여 넣으세요.\n"
                        f"저장 위치: {ENV_PATH}", classes="hint")
            yield Input(placeholder="sk-...", password=True, id="key")
            with Horizontal(id="buttons"):
                yield Button("취소", id="cancel")
                yield Button("저장", id="save", variant="primary")

    def on_mount(self):
        self.query_one("#key", Input).focus()

    def on_input_submitted(self, event: Input.Submitted):
        self.dismiss(event.value.strip() or None)

    def on_button_pressed(self, event: Button.Pressed):
        self.dismiss(self.query_one("#key", Input).value.strip() if event.button.id == "save" else None)

    def action_cancel(self):
        self.dismiss(None)


class Dashboard(TextualApp):
    TITLE = "Live to Subtitle"
    CSS = """
    Screen { layout: vertical; }
    #status { height: auto; border: round $accent; padding: 0 1; }
    #live { height: 3; border: round $secondary; padding: 0 1; }
    #subs { height: 1fr; border: round $primary; padding: 0 1; }
    #bottom { height: 10; }
    #stats { width: 44; border: round $secondary; padding: 0 1; }
    #log { width: 1fr; border: round $secondary; padding: 0 1; }
    """
    BINDINGS = [
        Binding("p", "pause", "일시정지"),
        Binding("plus,equals_sign", "delay(1)", "지연 +1초"),
        Binding("minus", "delay(-1)", "지연 -1초"),
        Binding("o", "open('/preview')", "미리보기"),
        Binding("l", "open('/log')", "로그 페이지"),
        Binding("c", "copy_url", "URL 복사"),
        Binding("k", "api_key", "API 키"),
        Binding("q", "quit", "종료"),
    ]

    def __init__(self, make_core, config_path: Path):
        super().__init__()
        self.make_core = make_core
        self.core: App | None = None
        self.config_path = config_path
        self.core_task: asyncio.Task | None = None
        self.gpu = ""
        self._partial_line = ""
        self.errors = 0

    def compose(self) -> ComposeResult:
        yield Static(id="status")
        yield Static(id="live")
        yield Static(id="subs")
        with Horizontal(id="bottom"):
            yield Static(id="stats")
            yield RichLog(id="log", max_lines=300, wrap=True, markup=False)
        yield Footer()

    async def on_mount(self):
        self.query_one("#status").border_title = "Live to Subtitle"
        self.query_one("#live").border_title = "지금 들리는 말"
        self.query_one("#subs").border_title = "최근 자막"
        self.query_one("#stats").border_title = "통계"
        self.query_one("#log").border_title = "로그"
        # 다른 모듈의 print 를 로그 창으로 (스레드에서 찍은 것 포함)
        self.begin_capture_print(self, stdout=True, stderr=True)
        # 용어집·발음 색인을 만드는 데 1~2초 걸리므로 화면을 먼저 띄우고 별도 스레드에서
        self.query_one("#live", Static).update(Text("준비 중…", style="grey50"))
        self.core = await asyncio.to_thread(self.make_core)
        self.core_task = asyncio.create_task(self.core.run())
        if not os.environ.get("OPENAI_API_KEY"):
            self.action_api_key()  # 키가 없으면 처음에 바로 물어봄
        self.core_task.add_done_callback(self._core_done)
        self.set_interval(0.1, self.refresh_fast)
        self.set_interval(0.5, self.refresh_slow)
        self.set_interval(3.0, self.poll_gpu)
        self.poll_gpu()
        self.refresh_slow()

    # ------------------------------------------------------------ 로그
    def on_print(self, event: events.Print):
        self._partial_line += event.text
        *lines, self._partial_line = self._partial_line.split("\n")
        log = self.query_one("#log", RichLog)
        for line in lines:
            line = line.rstrip()
            if not line or _SUB_LINE.match(line):
                continue
            bad = bool(_BAD.search(line))
            self.errors += bad
            log.write(Text(time.strftime("%H:%M:%S ") + line, style="red" if bad else ""))

    def _core_done(self, task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.core.status = "오류로 멈춤"
            self.query_one("#log", RichLog).write(Text(f"{type(exc).__name__}: {exc}", style="bold red"))

    # ------------------------------------------------------------ 화면 갱신
    def refresh_fast(self):
        """음량 막대와 지금 들리는 말 (0.1초마다)."""
        box = self.query_one("#live", Static)
        if self.core.paused:
            box.update(Text("일시정지 중 — P 를 누르면 다시 번역합니다", style="yellow"))
        else:
            live = [s for _, s in sorted(self.core.segments.items()) if s.state == "live" and s.zh]
            box.update(Text(live[-1].zh if live else "…", style="bold" if live else "grey50"))
        self.query_one("#status", Static).update(self._status())

    def refresh_slow(self):
        self.query_one("#subs", Static).update(self._subs())
        self.query_one("#stats", Static).update(self._stats())

    def _status(self) -> Table:
        c = self.core.cfg
        t = Table.grid(padding=(0, 2))
        t.add_column(style="grey62", no_wrap=True)
        t.add_column()
        state = "일시정지" if self.core.paused else self.core.status
        color = {"듣는 중": "green", "일시정지": "yellow"}.get(state, "red" if "오류" in state else "cyan")
        src = getattr(self.core, "source", None)
        name = getattr(src, "info", {}).get("name") if hasattr(src, "info") else Path(getattr(src, "path", "")).name
        level = self.core.segmenter.level if self.core.segmenter else 0.0
        inp = Text.assemble((f"● {state}", f"bold {color}"), "   ", (name or c["audio"]["source"], ""), "   ")
        inp.append_text(_meter(level))
        t.add_row("상태", inp)
        t.add_row("인식", f"{c['asr']['model']} / {c['asr']['device']}   {self.gpu}")
        refine = "켜짐" if c["translate"].get("refine", True) and c["server"].get("delay", 0) > 0 else "꺼짐"
        key = {"no_key": ("API 키 없음 — K 를 눌러 입력", "bold red"), "auth": ("API 키가 틀림 — K 를 눌러 다시 입력", "bold red"),
               "network": ("OpenAI 연결 실패", "red"), "model": ("모델 이름 확인 필요 (config.toml)", "red")}.get(
            self.core.key_status, ("", ""))
        tr = Text(f"{c['translate']['model']}   API {self.core.translator.calls}회   다시 번역 {refine}   ")
        tr.append(*key)
        t.add_row("번역", tr)
        n = len(self.core.clients)
        overlay = Text(self.core.base_url + "/  ", style="")
        overlay.append(f"연결 {n}개" if n else "연결 없음 (OBS 브라우저 소스를 확인하세요)", style="green" if n else "yellow")
        t.add_row("오버레이", overlay)
        delay = c["server"].get("delay", 0)
        t.add_row("방송 지연", f"{delay:g}초" + ("  (OBS 영상 지연·싱크 오프셋도 같은 값)" if delay else "  (자막을 받는 즉시 표시)"))
        return t

    def _subs(self) -> Table:
        t = Table(box=None, expand=True, show_header=True, header_style="grey62", padding=(0, 1))
        t.add_column("#", justify="right", width=5, style="grey50")
        t.add_column("중국어", ratio=2, style="grey62", no_wrap=True, overflow="ellipsis")
        t.add_column("한국어", ratio=3, no_wrap=True, overflow="ellipsis")
        t.add_column("", width=12, justify="right", no_wrap=True)
        height = max(1, self.query_one("#subs").size.height - 2)
        segs = [s for _, s in sorted(self.core.segments.items()) if s.state in ("final", "done", "skip")][-height:]
        for s in segs:
            tm = self.core.timing.get(s.id)
            if s.state == "final":
                mark = Text("번역 중", style="cyan")
            elif s.state == "skip":
                mark = Text("건너뜀", style="grey50")
            elif tm:
                mark = Text(("✓ " if tm["refined"] else "") + f"{tm['final'] - tm['end']:.1f}s",
                            style="green" if tm["refined"] else "")
            else:
                mark = Text("")
            ko = Text(s.ko or ("(번역 실패: 원문 표시)" if s.state == "done" else ""),
                      style="" if s.ko else "red")
            t.add_row(str(s.id), s.zh, ko, mark)
        return t

    def _stats(self) -> Table:
        t = Table.grid(padding=(0, 1))
        t.add_column(style="grey62")
        t.add_column()
        recent = list(self.core.timing.values())[-200:]
        lat = [r["final"] - r["end"] for r in recent]
        if lat:
            lat.sort()
            p90 = lat[min(len(lat) - 1, int(len(lat) * 0.9))]
            t.add_row("말 끝→번역", f"중앙 {statistics.median(lat):.1f}s  p90 {p90:.1f}s")
        else:
            t.add_row("말 끝→번역", "-")
        delay = self.core.cfg["server"].get("delay", 0)
        if delay and recent:
            on_time = sum(r["final"] <= r["start"] + delay for r in recent) / len(recent)
            t.add_row("제때 표시", f"{on_time * 100:.0f}%  (최근 {len(recent)}줄)")
        if self.core.refine_tried:
            t.add_row("다시 번역", f"{self.core.refine_applied}/{self.core.refine_tried} 반영")
        t.add_row("자막", f"{len(self.core.timing)}줄")
        t.add_row("오류", Text(str(self.errors), style="red" if self.errors else ""))
        return t

    @work(thread=True, exclusive=True)
    def poll_gpu(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.split("\n")[0]
            used, total, util = (int(x) for x in out.split(","))
            self.gpu = f"GPU {used / 1024:.1f}/{total / 1024:.0f}GB  {util}%"
        except (OSError, ValueError, subprocess.SubprocessError):
            self.gpu = ""

    # ------------------------------------------------------------ 키
    def action_pause(self):
        self.core.paused = not self.core.paused
        self.notify("일시정지: 새 문장을 번역하지 않습니다" if self.core.paused else "다시 번역합니다")

    async def action_delay(self, step: int):
        old = self.core.cfg["server"].get("delay", 0)
        new = max(0, min(20, old + step))
        if new == old:
            return
        await self.core.set_delay(new)
        saved = _save_delay(self.config_path, new)
        self.notify(f"방송 지연 {old:g}초 → {new:g}초" + (" (config.toml 저장)" if saved else "")
                    + f"\nOBS 영상 지연·싱크 오프셋도 {new * 1000:g}ms 로 바꾸세요.", timeout=6)

    def action_open(self, path: str):
        if self.core.base_url:
            webbrowser.open(self.core.base_url + path)

    def action_copy_url(self):
        url = self.core.base_url + "/"
        try:
            subprocess.run(["clip"], input=url.encode("ascii"), check=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            self.copy_to_clipboard(url)
        self.notify(f"복사했습니다: {url}")

    def action_api_key(self):
        async def done(key: str | None):
            if not key:
                return
            _save_env(ENV_PATH, "OPENAI_API_KEY", key)
            self.core.translator.set_api_key(key)
            self.core.key_status = status = await self.core.translator.warmup()
            msg = {"ok": "API 키를 저장했고 연결을 확인했습니다.", "auth": "키가 올바르지 않습니다. 다시 확인해 주세요.",
                   "network": "키는 저장했지만 OpenAI 에 연결하지 못했습니다 (인터넷 확인).",
                   "model": "키는 맞지만 config.toml 의 번역 모델 이름을 찾을 수 없습니다."}.get(status, status)
            self.notify(msg, severity="information" if status == "ok" else "error", timeout=8)

        self.push_screen(ApiKeyScreen(), done)

    async def action_quit(self):
        if self.core_task and not self.core_task.done():
            self.core_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.core_task
        self.exit()


def run(cfg: dict, config_path: str):
    Dashboard(lambda: App(cfg), Path(config_path)).run()
