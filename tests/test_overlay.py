"""OBS 오버레이(static/overlay.html) 를 헤드리스 브라우저(Edge 또는 Chrome)로 검증.

실행: .venv\\Scripts\\python.exe tests\\test_overlay.py
- 확정된 번역만, 한 줄만 표시 (임시 번역·스트리밍 중 번역은 숨김), 번역 실패 줄은 원문 표시
- 화면에 뜬 자막은 나중에 번역이 바뀌어도 그대로
- delay: '말 시작 + delay' 에 표시 (서버 시계가 달라도)
- 프로그램 재시작(세션 변경) 시 이전 자막 비움
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
from aiohttp import web

OVERLAY = Path(__file__).resolve().parents[1] / "live_to_subtitle" / "static" / "overlay.html"
PORT = 8931
SKEW = 100.0  # 서버 시계가 브라우저보다 100초 느리다고 가정
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    shutil.which("chrome") or "", shutil.which("chromium") or "", shutil.which("google-chrome") or "",
]

state = {"session": "s1", "delay": 0}
clients: set = set()


def server_now() -> float:
    return time.time() - SKEW


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    await ws.send_str(json.dumps({"t": "hello", "session": state["session"], "now": server_now(), "delay": state["delay"]}))
    async for _ in ws:
        pass
    clients.discard(ws)
    return ws


async def send(i, ko, state_="done", draft=False, spoken_at=0.0):
    m = {"t": "seg", "id": i, "zh": "中文", "ko": ko, "state": state_, "ko_draft": draft,
         "t_start": 0, "t_end": 0, "updated": server_now(), "spoken_at": spoken_at}
    for c in list(clients):
        await c.send_str(json.dumps(m))


async def main() -> int:
    browser = next((b for b in BROWSERS if b and os.path.exists(b)), None)
    if not browser:
        print("SKIP: Edge / Chrome 을 찾을 수 없습니다")
        return 0
    app = web.Application()
    app.router.add_get("/", lambda r: web.FileResponse(OVERLAY))
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()
    proc = subprocess.Popen([browser, "--headless=new", f"--remote-debugging-port={PORT + 1}",
                             f"--user-data-dir={tempfile.mkdtemp()}", "--window-size=1920,1080", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(50):
                try:
                    pages = await (await http.get(f"http://127.0.0.1:{PORT + 1}/json")).json()
                    break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.2)
            page = next(p for p in pages if p["type"] == "page")
            ws = await http.ws_connect(page["webSocketDebuggerUrl"], max_msg_size=0)
            n = [0]

            async def call(method, **params):
                n[0] += 1
                await ws.send_json({"id": n[0], "method": method, "params": params})
                while True:
                    m = await ws.receive_json()
                    if m.get("id") == n[0]:
                        return m.get("result", {})

            async def text():
                r = await call("Runtime.evaluate", expression="document.getElementById('subs').innerText")
                return r["result"].get("value", "").replace("\n", " / ")

            async def expect(name, want, wait=0.4):
                await asyncio.sleep(wait)
                got = await text()
                ok = got == want
                results.append(ok)
                print(f"{'OK  ' if ok else 'FAIL'} {name}: {got or '(빈 화면)'}" + ("" if ok else f"   (기대: {want or '(빈 화면)'})"))

            async def open_overlay(query=""):
                await call("Page.navigate", url=f"http://127.0.0.1:{PORT}/{query}")
                await asyncio.sleep(1.5)

            # 1) 확정 번역만, 한 줄만
            await open_overlay()
            await send(1, "첫째 확정")
            await expect("확정 번역 표시", "첫째 확정")
            await send(2, "둘째 임시", "live", True)
            await expect("임시 번역은 숨김", "첫째 확정")
            await send(2, "둘째 스트리밍", "final", True)
            await expect("번역 중인 문장은 숨김", "첫째 확정")
            await send(2, "둘째 확정")
            await expect("새 줄로 교체 (한 줄만)", "둘째 확정")
            await send(2, "둘째 다시 번역")
            await expect("화면에 뜬 자막은 그대로", "둘째 확정")
            await send(3, "")
            await expect("번역 실패는 원문", "中文")

            # 2) delay (서버 설정값 사용, 서버 시계가 100초 달라도)
            state.update(session="s2", delay=3)
            await open_overlay()
            await send(10, "지연 첫째", spoken_at=server_now())
            await expect("delay: 1초 뒤 아직", "", 1.0)
            await expect("delay: 3초 뒤 표시", "지연 첫째", 2.3)
            await send(11, "지연 둘째", spoken_at=server_now() - 1.0)
            await expect("delay: 앞 줄 유지", "지연 첫째", 1.0)
            await expect("delay: 2초 뒤 교체", "지연 둘째", 1.3)

            # 3) 재시작: 세션이 바뀌면 이전 자막 비움
            state.update(session="s3", delay=0)
            for c in list(clients):
                await c.close()
            await expect("재시작 후 이전 자막 비움", "", 2.5)
            await send(1, "새 실행 첫 자막")
            await expect("재시작 후 새 자막", "새 실행 첫 자막")
            await ws.close()
    finally:
        proc.terminate()
        await runner.cleanup()
    print("ALL OK" if all(results) else "SOME FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
