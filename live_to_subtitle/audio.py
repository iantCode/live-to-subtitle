"""오디오 입력: WASAPI loopback / 입력 장치 / 파일 -> 16kHz mono float32 청크를 큐로 전달."""
import queue
import threading
import time

import numpy as np
import soxr

SAMPLE_RATE = 16000


class _PyAudioSource:
    """PyAudioWPatch 콜백 스트림 공통 처리 (채널 평균 + 16kHz 리샘플)."""

    def __init__(self, out: queue.Queue, info: dict):
        import pyaudiowpatch as pyaudio

        self._pyaudio = pyaudio
        self.out = out
        self.info = info
        self.channels = int(info["maxInputChannels"])
        self.rate = int(info["defaultSampleRate"])
        self.resampler = soxr.ResampleStream(self.rate, SAMPLE_RATE, 1, dtype="float32")
        self.pa = None
        self.stream = None

    def _callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32)
        mono = data.reshape(-1, self.channels).mean(axis=1)
        chunk = self.resampler.resample_chunk(mono)
        if chunk.size:
            self.out.put(chunk)
        return (None, self._pyaudio.paContinue)

    def start(self):
        self.pa = self._pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=self._pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=int(self.info["index"]),
            frames_per_buffer=self.rate // 20,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def stop(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.pa is not None:
            self.pa.terminate()


def _find_device(device: str, loopback: bool) -> dict:
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        if loopback:
            if not device:
                return pa.get_default_wasapi_loopback()
            candidates = list(pa.get_loopback_device_info_generator())
        else:
            if not device:
                return pa.get_default_input_device_info()
            candidates = [
                pa.get_device_info_by_index(i)
                for i in range(pa.get_device_count())
                if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
                and not pa.get_device_info_by_index(i).get("isLoopbackDevice")
            ]
        for info in candidates:
            if device.lower() in info["name"].lower():
                return info
        raise RuntimeError(f"'{device}' 를 포함하는 장치를 찾지 못했습니다. --list-devices 로 확인하세요.")
    finally:
        pa.terminate()


class FileSource:
    """테스트용: 오디오/영상 파일을 디코딩해 (실시간 속도로) 흘려보냄."""

    def __init__(self, out: queue.Queue, path: str, realtime: bool = True):
        self.out = out
        self.path = path
        self.realtime = realtime
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.finished = threading.Event()
        self.t0 = 0.0

    def _run(self):
        audio = decode_file(self.path)
        step = SAMPLE_RATE // 10
        # 오디오 0초가 '재생'된 시각. 실시간 모드에서 청크는 그 소리가 끝난 시각에 들어감
        self.t0 = t0 = time.monotonic()
        for i in range(0, len(audio), step):
            if self._stop.is_set():
                break
            if self.realtime:
                delay = t0 + (i + step) / SAMPLE_RATE - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self.out.put(audio[i : i + step])
        self.finished.set()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()


def decode_file(path: str) -> np.ndarray:
    import av

    chunks = []
    with av.open(path) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def create_source(cfg: dict, out: queue.Queue):
    kind = cfg["source"]
    if kind == "file":
        if not cfg["file"]:
            raise RuntimeError("audio.source = 'file' 이면 audio.file 경로가 필요합니다.")
        return FileSource(out, cfg["file"], cfg["file_realtime"])
    if kind in ("loopback", "input"):
        info = _find_device(cfg["device"], loopback=(kind == "loopback"))
        print(f"[audio] {kind}: {info['name']} ({int(info['defaultSampleRate'])} Hz, {info['maxInputChannels']}ch)")
        return _PyAudioSource(out, info)
    raise RuntimeError(f"알 수 없는 audio.source: {kind}")


def list_devices():
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        print("== loopback (재생 장치의 소리) ==")
        for info in pa.get_loopback_device_info_generator():
            print(f"  {info['name']}")
        print("== input (마이크 / 라인 / 가상 케이블) ==")
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and info["hostApi"] == wasapi and not info.get("isLoopbackDevice"):
                print(f"  {info['name']}")
    finally:
        pa.terminate()
