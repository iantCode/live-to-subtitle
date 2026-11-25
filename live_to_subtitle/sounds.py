"""발음이 같은(비슷한) 게임 용어 찾기.

음성 인식은 고유명사를 같은 소리의 다른 글자로 적는 일이 잦다 (抖音 -> 斗音, 李白 -> 里白).
글자로는 용어집과 맞지 않으므로, 문장과 용어를 병음으로 바꿔 소리가 맞는 용어를 찾아 번역 모델에 힌트로 준다.

규칙 (잘못 찾는 일을 줄이기 위해 보수적으로)
- 성조는 무시하고, 흔히 섞이는 소리는 같게 본다: zh/z ch/c sh/s n/l, eng/en ing/in ang/an
- 용어 쪽은 여러 독음을 모두 고려한다 (茜 xī/qiàn, 泊 bó/pō)
- 인식된 글자는 jieba 단어 경계에 맞아야 하고 (我们今天 의 '们今' 은 안 봄), 흔한 단어 하나면 건너뜀 (一个, 今天)
- 2음절: 직접 고른 핵심 용어(config glossary)와 흔한 말이 아닌 캐릭터·고유명사 이름만, 소리가 완전히 같을 때
- 3음절: 소리가 완전히 같을 때 (핵심 용어는 4음절처럼) / 4음절 이상: 한 음절까지는 비슷한 소리(성모나 운모가 같음)로 달라도 됨
"""
import itertools
import re

import jieba
from pypinyin import Style, lazy_pinyin, pinyin

_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
             "y", "w")
_CJK = re.compile(r"[一-鿿]")
COMMON_FREQ = 30
MAX_HINTS = 6
# 용어집 2글자 이름을 찾을 때, 인식된 글자에 이보다 흔한 글자(你 有 说 会 里 吗 …)가 있으면 건너뜀 (jieba 빈도)
NAME_CHAR_FREQ = 20000
# 한 글자 기능어. 3음절 이상 구간이 이걸로 시작/끝나면 우연히 소리가 맞은 것 (是乐园 -> 失乐园)
FUNCTION_CHARS = set("是和的了在也就都还又与及或而但跟被把给对向从吗呢吧啊嘛呀着过地得")


def _norm(syl: str) -> tuple[str, str]:
    """음절 -> (성모, 운모) 정규화."""
    syl = syl.lower().replace("ü", "v")
    ini = next((i for i in _INITIALS if syl.startswith(i)), "")
    fin = syl[len(ini):]
    ini = {"zh": "z", "ch": "c", "sh": "s", "n": "l"}.get(ini, ini)
    for a, b in (("eng", "en"), ("ing", "in"), ("ang", "an")):
        if fin.endswith(a):
            fin = fin[: -len(a)] + b
    return ini, fin


def _key(sy: tuple[str, str]) -> str:
    return sy[0] + "-" + sy[1]


class SoundAlike:
    def __init__(self, terms: dict[str, str], core: set[str] = frozenset(), names: set[str] = frozenset()):
        """terms: 중국어 -> 한국어 (한자로만 된 2~8자 용어만 씀). core: 2음절도 찾을 핵심 용어.
        names: 용어집의 캐릭터·고유명사. 2음절은 그 자체가 흔한 말(扳手, 护盾)이 아닐 때만, 소리가 완전히 같을 때 찾음."""
        jieba.setLogLevel(60)
        jieba.initialize()
        self.freq = jieba.dt.FREQ
        self.terms = terms
        self.core = set(core)
        self.names2 = {cn for cn in names if len(cn) == 2 and not self._common(cn)}
        # 정확히 같은 소리: 음절 키 튜플 -> 용어들 / 한 음절이 다른 소리: 그 자리를 '*' 로 가린 튜플 -> (용어, 자리, 음절)
        self.exact: dict[tuple, set[str]] = {}
        self.masked: dict[tuple, list[tuple[str, int, tuple[str, str]]]] = {}
        for cn in terms:
            if not (2 <= len(cn) <= 8) or not all(_CJK.match(c) for c in cn) or cn[-1] in "的之了":
                continue
            if len(cn) == 2 and cn not in self.core and cn not in self.names2:
                continue
            # 소리가 완전히 같은지 볼 때는 여러 독음을 모두 (法露茜 의 茜 = xī)
            readings = pinyin(cn, style=Style.NORMAL, heteronym=True)
            for combo in itertools.islice(itertools.product(*readings), 8):
                keys = tuple(_key(_norm(s)) for s in combo)
                self.exact.setdefault(keys, set()).add(cn)
            # 한 음절이 달라도 되는 경우는 대표 독음만 (보조 독음까지 쓰면 우연히 맞는 게 너무 많음)
            # 3음절은 직접 고른 핵심 용어만 (伊斯尔 -> 伊瑟尔, 夜行主 -> 夜航主)
            if len(cn) >= 4 or (len(cn) == 3 and cn in self.core):
                syls = tuple(_norm(s) for s in lazy_pinyin(cn))
                keys = tuple(_key(s) for s in syls)
                for k in range(len(keys)):
                    masked = keys[:k] + ("*",) + keys[k + 1:]
                    self.masked.setdefault(masked, []).append((cn, k, syls[k]))

    def _common(self, word: str) -> bool:
        return self.freq.get(word, 0) >= COMMON_FREQ

    def find(self, text: str, exclude: set[str] = frozenset()) -> list[tuple[str, str, str]]:
        """문장에서 소리가 맞는 용어 -> [(용어, 한국어, 인식된 글자)] (글자가 똑같은 건 제외)."""
        found: dict[str, tuple[str, str, str]] = {}
        # 한자 덩어리별로 (영문·숫자·문장부호에서 끊음)
        for chunk in re.findall(r"[一-鿿]+", text):
            syls = [_norm(s) for s in lazy_pinyin(chunk)]
            keys = [_key(s) for s in syls]
            bounds, off, single, func = {0}, 0, set(), set()
            for tok in jieba.lcut(chunk):
                single.add((off, off + len(tok)))
                if tok in FUNCTION_CHARS:
                    func.add(off)  # 是 / 和 / 的 같은 한 글자 기능어 위치
                off += len(tok)
                bounds.add(off)
            for n in range(min(8, len(chunk)), 1, -1):
                for i in range(len(chunk) - n + 1):
                    heard = chunk[i:i + n]
                    if i not in bounds or i + n not in bounds:
                        continue  # 단어 중간에서 시작하거나 끝남
                    if (i, i + n) in single and self._common(heard):
                        continue  # 흔한 단어 하나 (一个, 今天, 环境 …)
                    if n >= 3 and (i in func or i + n - 1 in func):
                        continue  # 기능어로 시작하거나 끝남 (是乐园 -> 失乐园 같은 우연)
                    window = tuple(keys[i:i + n])
                    cands = set(self.exact.get(window, ()))
                    if n >= 3:
                        for k in range(n):
                            masked = window[:k] + ("*",) + window[k + 1:]
                            for cn, pos, syl in self.masked.get(masked, ()):
                                got = syls[i + k]
                                if pos == k and (got[0] == syl[0] or got[1] == syl[1]):
                                    cands.add(cn)
                    for cn in cands:
                        if cn == heard or cn in exclude or cn in found or cn in text:
                            continue
                        if n == 2 and cn not in self.core and any(self.freq.get(c, 0) >= NAME_CHAR_FREQ for c in heard):
                            continue  # 2글자 이름에 아주 흔한 글자가 섞임 (说你 -> 莎莉, 你有 -> 利奥 같은 우연)
                        found[cn] = (cn, self.terms[cn], heard)
                        if len(found) >= MAX_HINTS:
                            return list(found.values())
        return list(found.values())
