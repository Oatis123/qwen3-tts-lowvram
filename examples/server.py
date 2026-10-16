"""Low-VRAM Qwen3-TTS voice-clone server (FastAPI).

    python examples/server.py --ref-audio my_voice.wav --ref-text "Exactly what is said in it."

Endpoints:
  GET  /ping               health check
  POST /play               {"text": ...} queue a line for playback on a local audio device
  POST /stop               drop the queue and fade out what is playing
  GET  /is_playing         {"is_playing": bool}
  POST /synthesize         {"text": ...} -> WAV (24 kHz mono, 16-bit)
  POST /synthesize_stream  {"text": ...} -> raw s16le PCM chunks (24 kHz mono) as they're generated
"""
import argparse
import io
import logging
import os
import queue
import sys
import threading
import time
import wave
from contextlib import asynccontextmanager

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ref-audio", required=True, help="reference voice wav (clean speech, ~5-20 s)")
ap.add_argument("--ref-text", help="exact transcript of the reference")
ap.add_argument("--ref-text-file", help="read the transcript from a UTF-8 text file instead")
ap.add_argument("--language", default="English")
ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
ap.add_argument("--max-seq-len", type=int, default=512)
ap.add_argument("--chunk-size", type=int, default=8, help="codec frames per streamed chunk (12.5 = 1 s)")
ap.add_argument("--decode-context", type=int, default=25, help="0 = decode the whole reference every line")
ap.add_argument("--vram-cap", type=float, default=2.5, help="GB, 0 = no cap; raise it for 1.7B")
ap.add_argument("--output-device", default=None,
                help="sounddevice output for /play: index or part of the device name (default: system)")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=2027)
ap.add_argument("--hf-home", default=None, help="where to cache model weights (sets HF_HOME)")
args = ap.parse_args()

if args.hf_home:
    os.environ["HF_HOME"] = os.path.abspath(args.hf_home)
if args.ref_text_file:
    with open(args.ref_text_file, encoding="utf-8") as f:
        args.ref_text = f.read().strip()
if not args.ref_text:
    ap.error("--ref-text or --ref-text-file is required (voice cloning needs the transcript)")

import numpy as np
import sounddevice as sd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from faster_qwen3_tts import FasterQwen3TTS
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen3_tts_lowvram import optimize  # noqa: E402


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/is_playing" not in msg and "/ping" not in msg


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


def resolve_output_device(spec):
    if spec is None:
        return None
    if spec.isdigit():
        return int(spec)
    for i, dev in enumerate(sd.query_devices()):
        if spec.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
            return i
    raise SystemExit(f"No output device matching {spec!r}; see python -m sounddevice")


OUTPUT_DEVICE = resolve_output_device(args.output_device)


class StreamPlayer:
    """Feeds generated chunks to a sounddevice output stream, with click-free stop."""

    def __init__(self):
        self.q = queue.Queue()
        self.stream = None
        self.sr = 24000
        self.buffer = np.array([], dtype="float32")
        self.lock = threading.Lock()
        self.stopped = False

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            while not self.q.empty():
                self.buffer = np.append(self.buffer, self.q.get_nowait())
            if len(self.buffer) >= frames:
                outdata[:, 0] = self.buffer[:frames]
                self.buffer = self.buffer[frames:]
            else:
                available = len(self.buffer)
                if available > 0:
                    # ~8 ms fade when the buffer runs dry, so underruns don't click
                    fade_len = min(200, available)
                    self.buffer[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype="float32")
                    outdata[:available, 0] = self.buffer
                outdata[available:].fill(0)
                self.buffer = np.array([], dtype="float32")

    def __call__(self, chunk, sr):
        chunk = chunk.astype(np.float32, copy=False)
        with self.lock:
            if self.stopped:
                return
            self.q.put(chunk)
        if self.stream is None:
            self.sr = sr
            self.stream = sd.OutputStream(samplerate=sr, channels=1, callback=self._callback,
                                          dtype="float32", device=OUTPUT_DEVICE)
            self.stream.start()

    def clear(self):
        with self.lock:
            self.stopped = True
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            fade_len = min(int(0.05 * self.sr), len(self.buffer))
            if fade_len:
                self.buffer[:fade_len] *= np.linspace(1.0, 0.0, fade_len, dtype="float32")
            self.buffer = self.buffer[:fade_len]

    def resume(self):
        with self.lock:
            self.stopped = False

    def is_playing(self) -> bool:
        with self.lock:
            return not self.q.empty() or len(self.buffer) > 0

    def close(self):
        if self.stream is not None:
            while self.is_playing():
                time.sleep(0.05)
            self.stream.stop()
            self.stream.close()
            self.stream = None


model = None
player = None
text_queue = queue.Queue()
interrupt = threading.Event()
gpu_lock = threading.Lock()  # CUDA graphs + static KV cache: one generation at a time


def generate(text):
    with gpu_lock:
        try:
            yield from model.generate_voice_clone_streaming(
                text=text, language=args.language,
                ref_audio=args.ref_audio, ref_text=args.ref_text, chunk_size=args.chunk_size,
            )
        finally:
            torch.cuda.empty_cache()  # hand per-line scratch buffers back


def tts_worker():
    while (text := text_queue.get()) is not None:
        try:
            if not interrupt.is_set():
                player.resume()
                for chunk, sr, _ in generate(text):
                    if interrupt.is_set():
                        break
                    player(chunk, sr)
        except Exception as e:
            print(f"Worker error: {e}", file=sys.stderr)
        finally:
            text_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, player
    model = FasterQwen3TTS.from_pretrained(args.model, max_seq_len=args.max_seq_len)
    optimize(model, decode_context=args.decode_context or None, vram_cap_gb=args.vram_cap or None)
    player = StreamPlayer()
    for _ in generate("Warming up."):  # caches the voice prompt, captures CUDA graphs
        pass
    worker = threading.Thread(target=tts_worker, daemon=True)
    worker.start()
    yield
    text_queue.put(None)
    worker.join()
    player.close()


app = FastAPI(lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str


@app.get("/ping")
async def ping():
    return {"ok": True}


@app.get("/is_playing")
async def is_playing():
    return {"is_playing": not text_queue.empty() or (player is not None and player.is_playing())}


@app.post("/play")
async def play(payload: TTSRequest):
    interrupt.clear()
    text_queue.put(payload.text)
    return {"status": "enqueued", "queue_size": text_queue.qsize()}


@app.post("/stop")
async def stop():
    interrupt.set()
    while not text_queue.empty():
        try:
            text_queue.get_nowait()
            text_queue.task_done()
        except queue.Empty:
            break
    if player:
        player.clear()
    return {"status": "stopped"}


@app.post("/synthesize")
def synthesize(payload: TTSRequest):
    try:
        chunks, sr = [], 24000
        for chunk, sr, _ in generate(payload.text):
            chunks.append(chunk)
        if not chunks:
            raise HTTPException(status_code=500, detail="Model returned no audio")
        pcm = np.clip(np.concatenate(chunks) * 32767, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_stream")
def synthesize_stream(payload: TTSRequest):
    def pcm_chunks():
        for chunk, _, _ in generate(payload.text):
            yield np.clip(chunk * 32767, -32768, 32767).astype(np.int16).tobytes()

    return StreamingResponse(pcm_chunks(), media_type="application/octet-stream",
                             headers={"X-Audio-Format": "s16le", "X-Sample-Rate": "24000",
                                      "X-Channels": "1"})


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
