"""작은 단위 검증 (GPU·API 키 불필요).

실행: .venv\\Scripts\\python.exe tests\\test_units.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live_to_subtitle.asr import Corrections, clean_text  # noqa: E402
from live_to_subtitle.sounds import SoundAlike  # noqa: E402
from live_to_subtitle.translator import TermMatcher, one_line  # noqa: E402
from live_to_subtitle.tui import _save_delay, _save_env  # noqa: E402

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail and not ok else ""))


# 교정표: jieba 단어 경계에서만 바꿈
fix = Corrections("活属性 = 火属性\n# 메모\n斗音 = 抖音")
check("교정표 적용", fix("这是活属性的") == "这是火属性的", fix("这是活属性的"))
check("교정표: 다른 단어 안은 그대로", fix("生活属性") == "生活属性", fix("生活属性"))

# 인식 결과 정리
check("한 줄로", "\n" not in one_line("第一行\n第二行"))
check("clean_text 는 문자열", isinstance(clean_text("你好"), str))

# 용어집 파일: 문장에 나온 용어만, name 분류는 이름으로
terms = TermMatcher(str(ROOT / "glossary" / "example.tsv"))
found = dict(terms.find("今天在抖音直播间看到岚汐"))
check("용어집: 문장에 나온 용어", found.get("抖音") == "틱톡" and found.get("岚汐") == "란시", str(found))
check("용어집: name 분류", "岚汐" in terms.names and "抖音" not in terms.names, str(terms.names))

# 발음 힌트: 같은 소리의 다른 글자
sa = SoundAlike({"抖音": "틱톡", "岚汐": "란시", "赵云": "조운"}, core={"抖音"}, names={"岚汐", "赵云"})
hits = {cn: heard for cn, _, heard in sa.find("我昨天在斗音上看到")}
hits.update({cn: heard for cn, _, heard in sa.find("我觉得澜溪很强")})
check("발음 힌트: 핵심 용어", hits.get("抖音") == "斗音", str(hits))
check("발음 힌트: 이름", hits.get("岚汐") == "澜溪", str(hits))
# 아주 유명해 jieba 사전에 흔한 말로 올라 있는 이름(赵云)은 2글자 발음 찾기에서 뺌 (잘못 잡는 일 방지)
check("발음 힌트: 흔한 이름은 제외", "赵云" not in sa.names2)
check("발음 힌트: 흔한 글자 조합은 무시", not sa.find("你有没有说你"), str(sa.find("你有没有说你")))

# 설정 저장
with tempfile.TemporaryDirectory() as d:
    cfg = Path(d) / "config.toml"
    cfg.write_text('[audio]\nsource = "loopback"\n\n[server]\n# 주석\nport = 8765\ndelay = 10\n', encoding="utf-8")
    _save_delay(cfg, 8)
    text = cfg.read_text(encoding="utf-8")
    check("지연 저장", "delay = 8" in text and "# 주석" in text and 'source = "loopback"' in text, text)

    env = Path(d) / ".env"
    env.write_text("# 메모\nOTHER=1\nOPENAI_API_KEY=old\n", encoding="utf-8")
    _save_env(env, "OPENAI_API_KEY", "sk-new")
    text = env.read_text(encoding="utf-8")
    check(".env 저장", "OPENAI_API_KEY=sk-new" in text and "OTHER=1" in text and "old" not in text, text)

failed = [n for n, ok in results if not ok]
print("ALL OK" if not failed else f"SOME FAILED ({len(failed)})")
sys.exit(1 if failed else 0)
