"""스트리밍 문장 분할기.

0.4초마다 '아직 확정되지 않은 오디오 버퍼' 전체를 다시 인식하고, 아래 규칙으로 자막 단위를 확정한다.

  1. 침묵 확정   : VAD 기준으로 말이 silence_commit_sec 이상 끊기면 지금까지의 문장을 모두 확정
                   (길면 문장부호 기준으로 여러 줄로 나눔)
  2. 문장 끝     : 연속 두 번의 인식 결과가 똑같이 동의한 앞부분(안정 구간)에
                   。！？ 등이 있고 min_chars 이상이면 그 지점에서 확정
  3. 길이 초과   : 안정 구간이 soft_max_chars 를 넘으면 쉼표(，、：), 없으면 접속사(因为/所以/然后…) 앞에서 확정
  4. 강제 분할   : 전체가 max_chars / max_sec 를 넘으면 쉼표 > 접속사 > 가장 긴 단어 사이 쉼 > 글자 수 순으로 자름

모든 분할은 jieba 단어 경계에서만 일어난다 (转来转|去 처럼 단어 중간을 자르지 않음).

확정된 부분의 오디오는 버퍼에서 잘라내므로 다음 인식은 항상 '진행 중인 문장'만 다룬다.

인식 사이에도 0.1초마다 VAD(가벼움)로 말 끝을 살핀다.
  - 말이 끊긴 것이 보이면 그 시점까지 인식해 두고, 앱에 '말 멈춤' 신호를 보내 번역을 미리 시작하게 함
  - 침묵이 silence_commit_sec 에 이르면, 직전 인식이 이미 말 끝까지 다뤘으므로 다시 인식하지 않고 바로 확정
"""
import queue
import threading
import time
from collections.abc import Callable

import jieba
import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from .asr import ASR, Corrections, Word
from .audio import SAMPLE_RATE

jieba.setLogLevel(60)

STALE_SEC = 2.0
# 인식 사이에 VAD 로 말 끝을 확인하는 간격
VAD_CHECK_SEC = 0.1
# VAD 상 말 끝 뒤로 이만큼 조용하면 '말 멈춤' 으로 봄 (VAD 는 실제 침묵이 0.25초 이상일 때부터 끝을 알려줌)
PAUSE_SEC = 0.1
SENT_END = set("。！？；…")
_CONTEXT_PUNCT = str.maketrans({"？": "。", "！": "。"})
SOFT_BREAK = set("，、：")
# 이 단어 앞은 새 절이 시작되는 자연스러운 분할 지점
CONJUNCTIONS = {
    "然后", "所以", "因为", "但是", "可是", "不过", "而且", "并且", "或者", "如果",
    "虽然", "因此", "那么", "接下来", "另外", "其实", "比如", "于是", "同时", "结果",
}
# 이 단어로 끝나면 뒤에 말이 이어져야 하므로 그 뒤에서 자르지 않음 (…这样的一个|问题)
DANGLING = {
    "的", "地", "得", "之", "把", "被", "给", "在", "对", "跟", "和", "与", "及", "让", "从", "向",
    "往", "为", "为了", "关于", "通过", "比", "是", "去", "要", "会", "能", "可以", "想", "就", "也",
    "都", "还", "又", "再", "很", "太", "最", "更", "非常", "特别", "比较", "有点", "一直", "正在",
    "将", "这", "那", "这样", "那样", "这么", "那么", "这种", "那种", "这些", "那些", "哪", "每",
    "第", "一", "几", "某",
} | CONJUNCTIONS


PARTICLES = {"的", "了", "吗", "呢", "吧", "啊", "呀", "嘛", "啦", "的话", "着", "过", "地", "得"}
NOUN_CLASSIFIERS = set("个些种位款件只条张台部本名家场支首篇项份套批群辆块颗双座间所样类")


def _is_dangling(tok: str) -> bool:
    if tok in DANGLING or (len(tok) > 1 and tok.endswith("的")):
        return True
    # 一个 / 这个 / 一些 / 那款 / 三位 … (수량사 + 양사, 뒤에 명사가 와야 함)
    # 两天 / 三年 / 一次 처럼 시간·횟수 단위는 그 자체로 완결되므로 제외
    return len(tok) == 2 and tok[0] in "一二两三四五六七八九十几这那每某哪" and tok[1] in NOUN_CLASSIFIERS
_IGNORED = SENT_END | SOFT_BREAK | set("“”‘’\"'《》「」（）() -—·~")


def bare_text(text: str) -> str:
    """문장부호·공백을 뺀 글자만 (같은 문장인지 비교용)."""
    return "".join(c for c in text if c not in _IGNORED)


def char_len(text: str) -> int:
    """문장부호·공백을 뺀 글자 수."""
    return sum(1 for c in text if c not in _IGNORED)


def join_words(words: list[Word]) -> str:
    return "".join(w.text for w in words)


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Segmenter(threading.Thread):
    def __init__(
        self,
        audio_q: queue.Queue,
        asr: ASR,
        cfg: dict,
        asr_cfg: dict,
        on_partial: Callable[[int, str, bool, float], None],
        on_commit: Callable[[int, str, float, float, float], None],
    ):
        super().__init__(daemon=True)
        self.q = audio_q
        self.asr = asr
        self.cfg = cfg
        self.base_prompt = asr_cfg["initial_prompt"]
        # 교정된 문장이 화면·번역·다음 인식의 문맥(prompt)에 모두 쓰임
        self.correct = Corrections(asr_cfg.get("corrections", ""))
        self.on_partial = on_partial
        self.on_commit = on_commit
        self.stop_event = threading.Event()

        jieba.initialize()
        self.vad = VadOptions(
            threshold=cfg["vad_threshold"],
            min_silence_duration_ms=250,
            speech_pad_ms=100,
        )
        self.buf = np.zeros(0, np.float32)
        self.buf_t0 = 0.0  # buf[0] 의 스트림 시각 (초, 입력 시작 기준)
        self.recv_total = 0  # 지금까지 받은 샘플 수
        self.recv_wall = time.time()  # 마지막 오디오를 받은 시각 (epoch)
        self.level = 0.0  # 최근 입력 음량 (RMS)
        self.decoded_len = 0  # 마지막 인식이 다룬 버퍼 길이 (샘플)
        self.vad_checked_len = 0  # 마지막 VAD 확인 때의 버퍼 길이
        self.prev_texts: list[str] = []
        self.pending: list[Word] = []
        self.next_id = 1
        self.history: list[str] = []
        self.last_partial = ""
        self.last_paused = False

    # ------------------------------------------------------------------ loop
    def run(self):
        step = int(self.cfg["step_sec"] * SAMPLE_RATE)
        vad_step = int(VAD_CHECK_SEC * SAMPLE_RATE)
        last_audio = time.monotonic()
        while not self.stop_event.is_set():
            got = self._drain(timeout=0.05)
            now = time.monotonic()
            if got:
                last_audio = now
            try:
                if len(self.buf) - self.decoded_len >= step:
                    self._process()
                elif self.pending and len(self.buf) - self.vad_checked_len >= vad_step:
                    self._check_pause()
            except Exception as e:  # 인식 오류로 전체가 죽지 않도록
                print(f"[segmenter] error: {e!r}")
                self._reset_buffer()
            if self.pending and now - last_audio >= self._silence_needed(self.pending):
                # loopback 은 재생이 멈추면 데이터 자체가 안 들어오므로 시계 기준으로도 확정
                self._commit_all(self.pending)
                self._reset_buffer()

    def stop(self):
        self.stop_event.set()

    def flush(self):
        """입력이 끝났을 때 남은 문장을 확정."""
        if self.pending:
            self._commit_all(self.pending)
            self._reset_buffer()

    def _drain(self, timeout: float) -> bool:
        chunks = []
        try:
            chunks.append(self.q.get(timeout=timeout))
            while True:
                chunks.append(self.q.get_nowait())
        except queue.Empty:
            pass
        if chunks:
            self.buf = np.concatenate([self.buf, *chunks])
            self.recv_total += sum(len(c) for c in chunks)
            self.recv_wall = time.time()
            # 입력 음량 (TUI 표시용): 최근 오디오의 RMS
            last = chunks[-1]
            self.level = float(np.sqrt(np.mean(last * last))) if len(last) else 0.0
        return bool(chunks)

    # ------------------------------------------------------------ processing
    def _process(self):
        speech = get_speech_timestamps(self.buf, self.vad, sampling_rate=SAMPLE_RATE)
        if not speech:
            if self.pending:
                self._commit_all(self.pending)
            self._drop(len(self.buf) - int(0.3 * SAMPLE_RATE))
            self._reset_hypothesis()
            return

        # 앞쪽 무음은 버려서 인식 범위를 줄임
        lead = speech[0]["start"] - int(0.2 * SAMPLE_RATE)
        if lead > int(0.5 * SAMPLE_RATE) and not self.pending:
            self._drop(lead)
            speech = [{"start": s["start"] - lead, "end": s["end"] - lead} for s in speech]

        self.decoded_len = len(self.buf)
        words = self._dedupe_head(self.asr.transcribe(self.buf, self._prompt()))
        buf_sec = len(self.buf) / SAMPLE_RATE
        tail_silence = (len(self.buf) - speech[-1]["end"]) / SAMPLE_RATE

        if not words:
            if tail_silence >= self.cfg["silence_commit_sec"]:
                self._reset_buffer()
            return

        # 규칙 1: 침묵 확정 (VAD 상 말이 끝난 뒤에 시작하는 단어는 무음 구간 환각이므로 버림)
        if tail_silence >= self._silence_needed(words):
            self._commit_all(self._spoken(words, speech[-1]["end"]))
            self._drop(speech[-1]["end"])
            self._reset_hypothesis()
            return

        # 규칙 2~4: 안정 구간 안에서 자를 위치 찾기
        # 안정성은 글자만 비교 (Whisper 가 "了。" ↔ "了，" 처럼 문장부호만 바꾸는 경우가 잦음)
        texts = [bare_text(w.text) for w in words]
        stable_n = _common_prefix(self.prev_texts, texts)
        # 말해진 지 오래된 단어는 더 기다려도 문맥이 늘지 않으므로, 계속 흔들려도 지금 인식을 믿는다
        stale_n = sum(1 for w in words if w.end <= buf_sec - STALE_SEC)
        stable_n = max(stable_n, stale_n)
        i = 0
        while i < len(words):
            cut = self._find_cut(words, i, stable_n, buf_sec)
            if cut is None:
                break
            self._commit(words[i:cut])
            i = cut

        speech_end = speech[-1]["end"]
        if i > 0:
            n = int(self._boundary(words, i) * SAMPLE_RATE)
            self._drop(n)
            speech_end -= n
            shift = n / SAMPLE_RATE
            words = [Word(w.text, w.start - shift, w.end - shift) for w in words[i:]]

        self.pending = words
        self.prev_texts = [bare_text(w.text) for w in words]
        self.vad_checked_len = len(self.buf)
        if tail_silence >= PAUSE_SEC:
            # 말이 멈춤: 침묵 확정 때 나올 문장을 미리 알려 번역을 먼저 시작하게 함
            self._emit_partial(join_words(self._spoken(words, speech_end)), paused=True, start=words[0].start if words else None)
        else:
            self._emit_partial(join_words(words), start=words[0].start if words else None)

    def _check_pause(self):
        """인식 사이에 VAD 만으로 말 끝을 확인 (인식 결과 pending 이 있을 때)."""
        self.vad_checked_len = len(self.buf)
        speech = get_speech_timestamps(self.buf, self.vad, sampling_rate=SAMPLE_RATE)
        if not speech:
            return  # 다음 인식(_process)이 처리
        speech_end = speech[-1]["end"]
        tail_silence = (len(self.buf) - speech_end) / SAMPLE_RATE
        if tail_silence < PAUSE_SEC:
            return  # 아직 말하는 중
        if self.decoded_len < speech_end:
            # 마지막 인식이 말 끝까지 다루지 못함 -> 다음 인식 주기를 기다리지 않고 지금 인식
            self._process()
            return
        words = self._spoken(self.pending, speech_end)
        if tail_silence >= self._silence_needed(words):
            # 규칙 1 (침묵 확정): 마지막 인식이 이미 말 끝까지 다뤘으므로 다시 인식하지 않음
            self._commit_all(words)
            self._drop(speech_end)
            self._reset_hypothesis()
        else:
            self._emit_partial(join_words(words), paused=True, start=words[0].start if words else None)

    @staticmethod
    def _spoken(words: list[Word], speech_end: int) -> list[Word]:
        """VAD 상 말이 끝난 뒤에 시작하는 단어(무음 구간 환각)를 뺌. speech_end 는 버퍼 기준 샘플."""
        cutoff = speech_end / SAMPLE_RATE + 0.1
        return [w for w in words if w.start < cutoff]

    @staticmethod
    def _word_bounds(text: str) -> tuple[set[int], set[int], set[int]]:
        """jieba 단어 경계 (문자 오프셋), 접속사가 시작하는 오프셋, 뒤에 말이 이어져야 하는 경계."""
        bounds, conj, dangling = {0}, set(), set()
        off = 0
        for tok in jieba.lcut(text):
            if tok in CONJUNCTIONS:
                conj.add(off)
            if tok in PARTICLES:  # 앞 단어에 붙는 조사 앞에서는 자르지 않음 (…大乐卡|的那个)
                dangling.add(off)
            off += len(tok)
            bounds.add(off)
            if _is_dangling(tok):
                dangling.add(off)
        return bounds, conj, dangling

    def _find_cut(self, words: list[Word], i: int, stable_n: int, now: float) -> int | None:
        """words[i:] 에서 확정할 끝 위치(words 인덱스)를 찾음. now 는 버퍼 기준 현재 시각 (0 이면 시간 규칙 끔)."""
        c = self.cfg
        dur = now - words[i].start
        bounds, conj_starts, dangling = self._word_bounds(join_words(words[i:]))
        acc = off = 0
        soft = conj = clause = None
        gaps: list[tuple[float, int]] = []
        weak_gaps: list[tuple[float, int]] = []  # 구 중간 (…这样的一个|问题) - 최후의 수단
        # 인식 결과의 마지막 단어 뒤 문장부호는 Whisper 가 버퍼 끝에 습관적으로 붙이는 것이라
        # (예: "非常好。" 인데 실제로는 "好用" 을 말하는 중) 뒤에 단어가 더 있을 때만 믿는다.
        # 문장이 정말 끝났다면 침묵 확정(규칙 1)이 처리한다.
        for k in range(i, min(stable_n, len(words) - 1)):
            acc += char_len(words[k].text)
            off += len(words[k].text)
            if acc < c["min_chars"]:
                continue
            if acc > c["max_chars"]:
                break
            if off not in bounds:  # 단어 중간 (예: 转来转|去) 에서는 절대 자르지 않음
                continue
            last = words[k].text[-1]
            if last in SENT_END:
                return k + 1  # 규칙 2
            if last in SOFT_BREAK:
                soft = k + 1
            gap = (words[k + 1].start - words[k].end, k + 1)
            if off in dangling:
                weak_gaps.append(gap)
                continue
            if off in conj_starts:
                conj = k + 1
            gaps.append(gap)
            # 쉼표·짧은 쉼·접속사 앞이고, 그 지점이 말해진 지 clause_wait_sec 이 지났으면 후보
            natural = last in SOFT_BREAK or gap[0] >= c["pause_gap_sec"] or off in conj_starts
            if natural and now - words[k].end >= c["clause_wait_sec"]:
                clause = k + 1

        if clause:
            return clause  # 규칙 3a: 절 단위 확정 (가능한 가장 뒤 지점)

        stable_chars = sum(char_len(w.text) for w in words[i:stable_n])
        if stable_chars >= c["soft_max_chars"] and (soft or conj):
            return soft or conj  # 규칙 3: 쉼표, 없으면 접속사(因为/所以/然后…) 앞

        total_chars = sum(char_len(w.text) for w in words[i:])
        if total_chars >= c["max_chars"] or dur >= c["max_sec"]:  # 규칙 4
            if soft or conj:
                return soft or conj
            if gaps:
                return max(gaps)[1]
            # 길이 한계만 넘었으면 조금 더 기다려 본다 (말이 이어지면 더 나은 지점이 생김)
            if dur < c["max_sec"] and total_chars < c["max_chars"] * 1.3:
                return None
            if weak_gaps:
                return max(weak_gaps)[1]
            # 안정 구간이 없을 정도로 흔들리면 단어 경계에서 글자 수로 자름 (마지막 단어는 남김)
            acc = off = 0
            for k in range(i, len(words) - 1):
                acc += char_len(words[k].text)
                off += len(words[k].text)
                if acc >= c["soft_max_chars"] and off in bounds and off not in dangling:
                    return k + 1
            acc = off = 0
            for k in range(i, len(words) - 1):
                acc += char_len(words[k].text)
                off += len(words[k].text)
                if acc >= c["soft_max_chars"] and off in bounds:
                    return k + 1
        return None

    def _silence_needed(self, words: list[Word]) -> float:
        """침묵 확정에 필요한 시간. "…这样的一个" 처럼 말이 이어져야 하는 단어로 끝나면 더 기다린다."""
        base = self.cfg["silence_commit_sec"]
        text = join_words(words)
        if not text or text[-1] in SENT_END | SOFT_BREAK:
            return base
        toks = jieba.lcut(text)
        return max(base, 1.5) if toks and _is_dangling(toks[-1]) else base

    @staticmethod
    def _boundary(words: list[Word], i: int) -> float:
        """오디오를 자를 위치 (초).

        Whisper 단어 타임스탬프는 빠른 말에서 늦게 잡히는 경향이 있어, 다음 단어 시작보다
        넉넉히 앞에서 자른다. 그래서 겹친 글자가 다음 인식에 다시 나오면 _dedupe_head 가 지운다.
        """
        end = words[i - 1].end
        if i < len(words):
            end = min(end, words[i].start)
        return max(words[i - 1].start, end - 0.12)

    def _dedupe_head(self, words: list[Word]) -> list[Word]:
        """잘라낸 경계에 걸친 글자(최대 3자)가 다음 인식 맨 앞에 다시 나오는 경우 제거."""
        if not words or not self.history or words[0].start > 0.5:
            return words
        tail = "".join(ch for ch in self.history[-1] if ch not in _IGNORED)
        head = ""
        drop = 0
        for n, w in enumerate(words[:3], 1):
            head += "".join(ch for ch in w.text if ch not in _IGNORED)
            if len(head) > 3:
                break
            if head and tail.endswith(head):
                drop = n
        return words[drop:]

    def _prompt(self) -> str:
        # Whisper 는 prompt 의 문체를 따라 하므로, 한 번 생긴 버릇(모든 문장 끝에 ？ 등)이
        # 계속 이어지지 않도록 직전 문맥의 ？！ 는 。 로 바꿔서 넣는다
        context = "".join(self.history)[-60:].translate(_CONTEXT_PUNCT)
        return self.base_prompt + context

    # --------------------------------------------------------------- output
    def _commit_all(self, words: list[Word]):
        """침묵 확정: 길면 문장부호 기준으로 여러 줄로 나눠서 확정."""
        i = 0
        while i < len(words):
            cut = self._find_cut(words, i, len(words), now=0.0)
            if cut is None or cut >= len(words):
                break
            self._commit(words[i:cut])
            i = cut
        if i < len(words):
            self._commit(words[i:])

    def _commit(self, words: list[Word]):
        text = self.correct(join_words(words))
        if char_len(text) == 0:
            return
        seg_id = self.next_id
        self.next_id += 1
        self.history.append(text)
        self.history = self.history[-5:]
        self.last_partial = ""
        self.last_paused = False
        t_start = self.buf_t0 + words[0].start
        self.on_commit(seg_id, text, t_start, self.buf_t0 + words[-1].end, self._wall(t_start))

    def _emit_partial(self, text: str, paused: bool = False, start: float | None = None):
        """paused: 말이 멈춘 상태라 이 문장이 곧 그대로 확정될 가능성이 큼. start: 첫 단어 시작 (버퍼 기준 초)."""
        text = self.correct(text)
        if text != self.last_partial or paused != self.last_paused:
            self.last_partial, self.last_paused = text, paused
            spoken_at = self._wall(self.buf_t0 + start) if start is not None else 0.0
            self.on_partial(self.next_id, text, paused, spoken_at)

    def _wall(self, t: float) -> float:
        """스트림 시각(입력 시작 기준 초) -> 그 소리가 들어온 실제 시각 (epoch 초).
        마지막으로 받은 오디오를 기준으로 계산하므로 장시간 실행해도 사운드카드 시계 오차가 쌓이지 않음."""
        return self.recv_wall - (self.recv_total / SAMPLE_RATE - t)

    def _reset_hypothesis(self):
        self.prev_texts = []
        self.pending = []
        self._emit_partial("")

    def _drop(self, n: int):
        """버퍼 앞쪽 n 샘플을 버림 (스트림 시각 유지)."""
        n = max(0, min(n, len(self.buf)))
        self.buf = self.buf[n:]
        self.buf_t0 += n / SAMPLE_RATE
        self.decoded_len = len(self.buf)
        self.vad_checked_len = len(self.buf)

    def _reset_buffer(self):
        self._drop(len(self.buf))
        self._reset_hypothesis()
