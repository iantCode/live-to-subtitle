"""faster-whisper 래퍼: 버퍼 전체를 디코딩해 단어(문장부호 포함) + 타임스탬프 목록을 돌려줌."""
import glob
import importlib.util
import os
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

# Whisper 가 무음/음악 구간에서 자주 지어내는 문구
HALLUCINATIONS = (
    "字幕由",
    "字幕提供",
    "Amara.org",
    "请不吝点赞",
    "订阅 转发",
    "打赏支持",
    "明镜与点点",
    "谢谢观看",
    "感谢观看",
    "以下是普通话",
    "使用简体中文",
)

# 반각 문장부호 -> 전각 (문장 자르기 규칙을 단순하게 유지)
_PUNCT_MAP = str.maketrans({",": "，", "?": "？", "!": "！", ";": "；", ":": "："})

_FULLWIDTH_ALNUM = "０-９Ａ-Ｚａ-ｚ"
# Whisper 가 가끔 문장부호 자리에 전각 숫자/영문 한 글자를 넣음 (不停７现在是 / 还好吧Ｂ有好多).
# 진짜 전각 숫자는 두 글자 이상 붙어 나오므로 (２００), 홀로 있는 한 글자만 쉼표로 바꾼다.
_LONE_FULLWIDTH = re.compile(rf"(?<![{_FULLWIDTH_ALNUM}])[{_FULLWIDTH_ALNUM}](?![{_FULLWIDTH_ALNUM}])")
# 자막식 대화 대시 (-是吗？-当然可以) : 번역에 그대로 옮겨져 두 줄 자막이 됨
_DIALOGUE_DASH = re.compile(r"(^|[，。！？；：])\s*[-－—]+\s*")


class Corrections:
    """음성 인식이 반복해서 잘못 적는 말을 바로잡는 교정표 (config [asr] corrections, 한 줄에 '틀린 글자 = 맞는 글자').

    긴 것부터 바꾸고, 다른 단어의 일부인 경우(生活属性 안의 活属性)는 건너뛰도록
    jieba 단어 경계와 맞는 위치에서만 바꾼다.
    """

    def __init__(self, text: str):
        pairs = []
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if "=" not in line:
                continue
            wrong, right = (part.strip() for part in line.split("=", 1))
            if wrong and right and wrong != right:
                pairs.append((wrong, right))
        self.pairs = sorted(pairs, key=lambda p: -len(p[0]))

    def __call__(self, text: str) -> str:
        for wrong, right in self.pairs:
            if wrong in text:
                text = self._replace(text, wrong, right)
        return text

    @staticmethod
    def _replace(text: str, wrong: str, right: str) -> str:
        import jieba

        bounds, off = {0}, 0
        for tok in jieba.lcut(text):
            off += len(tok)
            bounds.add(off)
        out, i = [], 0
        while (j := text.find(wrong, i)) != -1:
            if j in bounds and j + len(wrong) in bounds:
                out.append(text[i:j] + right)
            else:
                out.append(text[i:j + len(wrong)])
            i = j + len(wrong)
        return "".join(out) + text[i:]


def _leak_pattern(terms: list[str]) -> "re.Pattern | None":
    """hotwords 누출: 같은 용어가 (한 글자 이하 사이를 두고) 세 번 이상 반복되거나, 용어 셋 이상이 붙어 나옴.
    (사람이 실제로 "月卡，月卡" 처럼 두 번 말하는 건 남김)"""
    if not terms:
        return None
    alt = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"({alt})(?:[^\w]?.?[^\w]?\1){{2,}}|(?:(?:{alt})[^\w]?){{3,}}")


def clean_text(text: str) -> str:
    """인식 결과에 끼는 엉뚱한 글자 정리 (전각 숫자/영문 대체 문장부호, 대화 대시)."""
    text = _LONE_FULLWIDTH.sub("，", text)
    text = "".join(unicodedata.normalize("NFKC", c) if "０" <= c <= "ｚ" and c.isalnum() else c for c in text)
    return _DIALOGUE_DASH.sub(r"\1", text)


@dataclass
class Word:
    text: str
    start: float
    end: float


def add_cuda_dll_dirs():
    """pip 로 설치한 nvidia-cublas / nvidia-cudnn 의 DLL 을 찾을 수 있게 함 (Windows)."""
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    for base in spec.submodule_search_locations:
        for bin_dir in glob.glob(os.path.join(base, "*", "bin")):
            os.add_dll_directory(bin_dir)
            # CTranslate2 는 cuBLAS 를 지연 로드하므로 PATH 에도 넣어야 함
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


class ASR:
    def __init__(self, cfg: dict):
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        self.cfg = cfg
        print(f"[asr] loading {cfg['model']} on {cfg['device']} ({cfg['compute_type']}) ...")
        self.model = WhisperModel(
            cfg["model"],
            device=cfg["device"],
            compute_type=cfg["compute_type"],
            download_root=cfg["download_root"] or None,
        )
        self._t2s = None
        if cfg["to_simplified"]:
            import opencc

            self._t2s = opencc.OpenCC("t2s")
        self.hotwords = " ".join(cfg.get("hotwords", "").split())
        self._leak = _leak_pattern(self.hotwords.split())

    def normalize(self, text: str) -> str:
        text = clean_text(text.strip().translate(_PUNCT_MAP))
        if self._t2s is not None:
            text = self._t2s.convert(text)
        return text

    def transcribe(self, audio: np.ndarray, prompt: str) -> list[Word]:
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg["language"],
            beam_size=self.cfg["beam_size"],
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=prompt or None,
            hotwords=self.hotwords or None,
        )
        words: list[Word] = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                continue
            seg_words = [Word(t, float(w.start), float(w.end)) for w in seg.words or [] if (t := self.normalize(w.word))]
            kept = self._cut_leak(seg_words)
            if len(kept) < len(seg_words):  # hotwords 를 받아 적은 부분을 잘라냄 (앞부분은 진짜 말)
                words.extend(kept)
                break  # 이후 구간도 같은 환각이 이어지므로 버림
            if seg.compression_ratio > 2.4:  # 같은 말 반복 = 환각
                continue
            if any(h in seg.text for h in HALLUCINATIONS):
                continue
            words.extend(seg_words)
        return words

    def _cut_leak(self, words: list[Word]) -> list[Word]:
        """Whisper 가 hotwords 목록을 받아 적은 부분(用语A的用语A / 用语B用语C用语B…)부터 잘라냄."""
        if self._leak is None or not words:
            return words
        text, owner = "", []
        for i, w in enumerate(words):
            text += w.text
            owner += [i] * len(w.text)
        m = self._leak.search(text)
        return words if m is None else words[: owner[m.start()]]
