"""OpenAI (GPT-6 Luna) 로 중국어 자막 한 줄을 한국어로 번역 (스트리밍)."""
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import openai

SKIP = "<skip>"


def one_line(text: str) -> str:
    """자막은 한 줄: 모델이 대화처럼 줄을 나눠도 공백으로 이어 붙임."""
    return " ".join(text.split())


def strip_skip(text: str) -> str:
    """<skip> 표시를 지움. 모델이 번역 뒤에 붙이는 경우("…요. <skip>")와 스트리밍 도중 잘린 "<sk" 도 처리."""
    text = text.replace(SKIP, "")
    for n in range(len(SKIP) - 1, 0, -1):
        if text.endswith(SKIP[:n]):
            text = text[:-n]
            break
    return one_line(text)


class TermMatcher:
    """용어집 파일(CN \\t KR) 에서 문장에 실제로 나온 용어만 찾음 (왼쪽부터, 가장 긴 용어 우선)."""

    MAX_TERMS = 20
    # 파일 형식: 한 줄에 `중국어<TAB>한국어[<TAB>분류]`, '#' 줄은 메모.
    # 분류가 이름이면 2글자여도 발음 힌트로 찾음 (물건·UI 용어의 2글자는 흔한 말과 겹치기 쉬워 찾지 않음)
    NAME_CATEGORIES = {"name", "person", "character", "speaker", "npc", "place", "encyclopedia"}

    def __init__(self, path: str):
        self.by_first: dict[str, list[tuple[str, str]]] = {}
        self.names: set[str] = set()
        self.size = 0
        if not path:
            return
        if not Path(path).exists():
            print(f"[translate] 경고: 용어집 파일 '{path}' 이 없습니다.")
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or "\t" not in line:
                    continue
                parts = line.rstrip("\n").split("\t")
                cn, kr = parts[:2]
                if cn and kr:
                    self.by_first.setdefault(cn[0], []).append((cn, kr))
                    self.size += 1
                    if len(parts) > 2 and parts[2].strip().lower() in self.NAME_CATEGORIES:
                        self.names.add(cn)
        for terms in self.by_first.values():
            terms.sort(key=lambda t: -len(t[0]))

    def find(self, text: str) -> list[tuple[str, str]]:
        found: dict[str, str] = {}
        i = 0
        while i < len(text) and len(found) < self.MAX_TERMS:
            for cn, kr in self.by_first.get(text[i], ()):
                if text.startswith(cn, i):
                    found.setdefault(cn, kr)
                    i += len(cn)
                    break
            else:
                i += 1
        return list(found.items())

INSTRUCTIONS = """You translate live Mandarin Chinese broadcast speech into Korean subtitles shown on stream in real time.

The Chinese text comes from automatic speech recognition, so it may contain homophone errors, missing punctuation, or a sentence cut off in the middle. Use the preceding lines for context and silently fix obvious recognition errors.

Rules:
- Output only the Korean subtitle text on a single line: no quotes, labels, dialogue dashes, romanization, or notes.
- Keep it short and natural for on-screen reading; match the speaker's tone (casual stays casual).
- Translate fragments as fragments; never invent content that was not said.
- Do not repeat content that belongs to the previous lines.
- Follow the glossary for names and terms. Speech recognition often writes glossary terms with similar-sounding wrong characters; when the context fits, translate them as the glossary term. But do not force a glossary term onto an ordinary word that already makes sense in the sentence.
- If the input is only noise, filler, or not meaningful speech, output exactly <skip>."""


class Translator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # OPENAI_API_KEY 는 환경변수 (.env) 에서 읽음
        self.client = None
        self.calls = 0  # 지금까지 보낸 번역 요청 수 (TUI 표시용)
        if os.environ.get("OPENAI_API_KEY"):
            self.client = openai.AsyncOpenAI(timeout=20.0, max_retries=1)
        else:
            print("[translate] 경고: OPENAI_API_KEY 가 없어 번역을 끕니다 (.env 에 추가하세요). 인식만 동작합니다.")
        self.history: dict[int, tuple[str, str]] = {}
        self.style = cfg["style"].strip()
        # config 의 glossary (항상 넣음) + 용어집 파일 (문장에 나온 용어만 넣음)
        # '#' 로 시작하는 줄은 메모 (모델에 보내지 않음)
        self.glossary = [line.strip() for line in cfg["glossary"].splitlines()
                         if line.strip() and not line.strip().startswith("#")]
        self.terms = TermMatcher(cfg.get("glossary_file", ""))
        if self.terms.size:
            print(f"[translate] 용어집 {cfg['glossary_file']}: {self.terms.size}개")
        # 음성 인식이 같은 소리의 다른 글자로 적은 용어 찾기 (斗音 -> 抖音)
        self.sounds = None
        if cfg.get("sound_alike", True):
            from .sounds import SoundAlike

            vocab = {cn: kr for terms in self.terms.by_first.values() for cn, kr in terms}
            core = set()
            for line in self.glossary:
                if "=" in line and line.split("=")[0].split():
                    cn = line.split("=")[0].split()[0]
                    vocab[cn] = line.split("=", 1)[1].strip()
                    core.add(cn)
            self.sounds = SoundAlike(vocab, core, names=self.terms.names)

    def _extra(self, zh: str) -> str:
        parts = []
        if self.style:
            parts.append(f"<broadcast_style>\n{self.style}\n</broadcast_style>")
        # "抖音 (dǒu yīn) = 틱톡" 처럼 발음을 붙인 줄도 있으므로 첫 단어만 비교
        fixed = {line.split("=")[0].split()[0] for line in self.glossary if line.split("=")[0].strip()}
        matched = [(cn, kr) for cn, kr in self.terms.find(zh) if cn not in fixed]
        lines = self.glossary + [f"{cn} = {kr}" for cn, kr in matched]
        if self.sounds is not None:
            # 고정 용어(抖音 등)도 '이 문장의 斗音 이 그것' 이라고 짚어 주는 게 핵심이라 빼지 않음
            exclude = {cn for cn, _ in matched}
            lines += [f"{cn} = {kr}  (the line probably has it as '{heard}', a speech-recognition error)"
                      for cn, kr, heard in self.sounds.find(zh, exclude)]
        if lines:
            parts.append("<glossary>\n" + "\n".join(lines) + "\n</glossary>")
        return "\n\n".join(parts)

    def set_api_key(self, key: str):
        """실행 중에 API 키를 바꿈 (TUI 에서 입력한 경우)."""
        os.environ["OPENAI_API_KEY"] = key
        self.client = openai.AsyncOpenAI(timeout=20.0, max_retries=1) if key else None

    async def warmup(self) -> str:
        """시작할 때 연결을 미리 열어 첫 자막의 지연(DNS/TLS)을 없애고, 키·모델 이름을 확인.
        결과: "ok" / "no_key" / "auth" (키가 틀림) / "model" (모델 이름 없음) / "network"."""
        if self.client is None:
            return "no_key"
        result = "ok"
        models = [self.cfg["model"]]
        if self.cfg.get("draft_model") and self.cfg["draft_model"] != self.cfg["model"]:
            models.append(self.cfg["draft_model"])
        for model in models:
            try:
                await self.client.models.retrieve(model)
                print(f"[translate] {model} 연결 확인")
            except openai.AuthenticationError:
                print("[translate] 인증 실패: .env 의 OPENAI_API_KEY 를 확인하세요.")
                return "auth"
            except openai.NotFoundError:
                print(f"[translate] 모델 '{model}' 을 찾을 수 없습니다. config.toml 을 확인하세요.")
                result = "model"
            except (openai.APIStatusError, openai.APIConnectionError) as e:
                print(f"[translate] 연결 확인 실패: {e!r}")
                return "network"
        return result

    def remember(self, seg_id: int, zh: str, ko: str):
        self.history[seg_id] = (zh, ko)
        # 뒤늦게 다시 번역하는 줄(refine)도 자기 앞 문맥을 볼 수 있도록 몇 줄 더 보관
        for old in sorted(self.history)[: -(self.cfg["context_segments"] + 8)]:
            del self.history[old]

    def _input(self, seg_id: int, zh: str, draft: bool, following: str = "") -> str:
        before = [(i, z, k) for i, (z, k) in sorted(self.history.items()) if i < seg_id][-self.cfg["context_segments"]:]
        lines = [f"{z}\n→ {k}" for _, z, k in before]
        parts = []
        extra = self._extra(zh)
        if extra:
            parts.append(extra)
        if lines:
            parts.append("<previous_lines>\n" + "\n".join(lines) + "\n</previous_lines>")
        note = " (the speaker is still talking; this line may be cut off mid-sentence)" if draft else ""
        parts.append(f"Translate this line{note}:\n<line>{zh}</line>")
        if following:
            # 방송을 늦춰 자막을 말 시작에 맞추는 경우, 다음 줄이 나온 뒤 다시 번역할 시간이 있음
            parts.append("<following_lines>\n" + following + "\n</following_lines>\n"
                         "(What the speaker said right after this line. Use it only to understand this line, "
                         "e.g. to resolve a sentence cut off mid-way or a mis-recognized word. Translate only <line>.)")
        return "\n\n".join(parts)

    async def translate(
        self,
        seg_id: int,
        zh: str,
        draft: bool,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        following: str = "",
    ) -> str | None:
        """번역 결과를 돌려줌. 건너뛸 내용이면 "", 실패하면 None."""
        if self.client is None:
            return None
        self.calls += 1
        text = ""
        try:
            stream = await self.client.responses.create(
                # 말하는 도중의 임시 번역은 곧 바뀌므로 더 빠른 모델을 쓸 수 있음 (draft_model)
                model=(self.cfg.get("draft_model") or self.cfg["model"]) if draft else self.cfg["model"],
                instructions=INSTRUCTIONS,
                input=self._input(seg_id, zh, draft, following),
                reasoning={"effort": self.cfg["effort"]},
                max_output_tokens=1024,
                store=False,
                stream=True,
            )
            async for event in stream:
                if event.type == "response.output_text.delta":
                    text += event.delta
                    visible = strip_skip(text)
                    if on_text and visible:
                        await on_text(visible)
                elif event.type in ("response.failed", "error"):
                    print(f"[translate] 응답 실패 #{seg_id}: {event}")
                    return None
        except openai.AuthenticationError:
            print("[translate] 인증 실패: .env 의 OPENAI_API_KEY 를 확인하세요.")
            return None
        except openai.RateLimitError:
            print("[translate] 요청 한도 초과 (429)")
            return None
        except openai.APIStatusError as e:
            print(f"[translate] API 오류 {e.status_code}: {e.message}")
            return None
        except openai.APIConnectionError as e:
            print(f"[translate] 연결 오류: {e!r}")
            return None

        return strip_skip(text)  # "" = 건너뜀
