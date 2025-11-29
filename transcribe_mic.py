import argparse
import queue
import sys
import time

import sounddevice as sd
import threading
from faster_whisper import WhisperModel
import numpy as np
import webrtcvad
from dataclasses import dataclass

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = FRAME_MS * SAMPLE_RATE // 1000
OVERLAPPING_SECONDS = 0.30
OVERLAP_SAMPLES = int(OVERLAPPING_SECONDS * SAMPLE_RATE)
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)
ROLLING_CONTEXT_SECONDS = 0.50
ROLLING_SAMPLES = int(ROLLING_CONTEXT_SECONDS * SAMPLE_RATE)
MAX_SILENCE_MS = 400
VAD_AGGRESIVENESS = 2

assert SAMPLE_RATE in (8000, 16000, 32000, 48000)
assert FRAME_MS in (10, 20, 30)
assert FRAME_SAMPLES * 1000 // SAMPLE_RATE == FRAME_MS  # sanity check

@dataclass
class Metrics:
    audio_sec: float = 0.0
    compute_sec: float = 0.0
    chunks: int = 0

def float_to_int(val: float) -> int:
    val = np.clip(val, -1.0, 1.0)
    return (val * 2 ** 15).astype(np.int16)

def int_to_float(val: int) -> float:
    return val / 2 ** 15

class MicStream:
    """Pulls 16kHz mono float32 frames from the default mic."""
    def __init__(self):
        self.q = queue.Queue(maxsize=100)

    def callback(self, indata, num_frames, time_info, status): # to put into the queue
        if status:
            print(status, file=sys.stderr)
        if indata.ndim == 2:
            data = np.mean(indata, axis=1)
        else:
            data = indata
        self.q.put_nowait(data)
    def __enter__(self): # to start the stream
        self.stream = sd.InputStream(channels = 1, samplerate = SAMPLE_RATE, blocksize = FRAME_SAMPLES, dtype = 'float32', callback= self.callback)
        self.stream.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop()
        self.stream.close()
    def read(self):
        # Blocking read of one blocksize frame(~30ms)
        return self.q.get()

class VADChunker:
    """Assemble voice audio into 1 s chunk and emit partial while speacking and full while silence"""

    def __init__(self):
        aggresiveness = VAD_AGGRESIVENESS
        self.vad = webrtcvad.Vad(aggresiveness)
        self.silence_ms = 0 # Time duraiton for which silence is seen
        self.voiced_buffer = np.zeros(0, dtype = np.float32) # empty array -> no samples
        self.last_final_tail = np.zeros(0, dtype = np.float32) # voiced tail of the last segment

    def is_speech(self, frame_f32):
        if frame_f32.ndim != 1 or frame_f32.size != FRAME_SAMPLES:
            return False
        if not np.isfinite(frame_f32).all():
            frame_f32 = np.nan_to_num(frame_f32, nan=0.0, posinf=0.0, neginf=0.0)
        pcm16 = float_to_int(frame_f32)
        pcm_bytes = pcm16.tobytes()
        return self.vad.is_speech(pcm_bytes, sample_rate = SAMPLE_RATE)

    def push_and_maybe_yield(self, frame_f32):
        """
        :param frame_f32:
        :return: Yield tuples: partial- in between speech at 1 sec gaps , final - when there is a long silence
        """

        out = []
        if self.is_speech(frame_f32):
            self.silence_ms = 0 # reset silence
            self.voiced_buffer = np.concatenate([self.voiced_buffer, frame_f32])

            if len(self.voiced_buffer) >= CHUNK_SAMPLES:
                ctx = self.last_final_tail[-ROLLING_SAMPLES:]
                audio = np.concatenate([ctx, self.voiced_buffer])
                out.append(("partial", audio)) # output every chunk samples

                keep = self.voiced_buffer[-OVERLAP_SAMPLES:]
                self.voiced_buffer = keep

        else:
            self.silence_ms += FRAME_MS
            if self.silence_ms >= MAX_SILENCE_MS:
                ctx = self.last_final_tail[-ROLLING_SAMPLES:]
                audio = np.concatenate([ctx, self.voiced_buffer])
                if len(audio) > ROLLING_SAMPLES:
                    out.append(("final", audio))
                    self.last_final_tail = audio[-ROLLING_SAMPLES:].copy()
                self.voiced_buffer = np.zeros(0, dtype = np.float32)
                self.silence_ms = 0

        return out



def asr_transcribe(model, audio_f32, language='en'):
    t0 = time.time()
    segments, info = model.transcribe(audio_f32, language = language, beam_size=1, vad_filter = False, suppress_blank = True, temperature = 0.0, condition_on_previous_text = False, word_timestamps= False)
    text = "".join(seg.text for seg in segments).strip()
    comp = time.time() - t0
    return text, comp, info.duration


def transcription_worker(model, q, metrics):
    """A worker thread that pulls audio chunks from a queue and transcribes them."""
    while True:
        try:
            # Get audio chunk and tag from the queue
            audio_f32, tag = q.get()

            # Check for shutdown signal
            if audio_f32 is None:
                break

            # --- This is the slow part ---
            text, comp, dur = asr_transcribe(model, audio_f32)
            # ----------------------------

            # Update metrics (thread-safe since it's just incrementing)
            metrics.compute_sec += comp
            metrics.audio_sec += len(audio_f32) / SAMPLE_RATE
            metrics.chunks += 1

            # Print the result
            if tag == "partial":
                if text:
                    print("[partial] " + text, flush=True)
            else:
                if text:
                    print("[final] " + text, flush=True)

            q.task_done()
        except Exception as e:
            print(f"Error in transcription worker: {e}")
            q.task_done()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default= "small", help="Path to model .wav file.")
    args = parser.parse_args()

    device = "cpu"

    print(f"[init] Loading Faster-Whisper model : {args.model}")
    model = WhisperModel(args.model, device = device, compute_type = "int8")

    metrics = Metrics()
    vad = VADChunker()

    transcription_q = queue.Queue()

    worker_thread = threading.Thread(
        target=transcription_worker,
        args=(model, transcription_q, metrics),
        daemon=True  # Thread will exit when main program exits
    )
    worker_thread.start()

    def handle_audio_block(audio_f32, tag):
        nonlocal metrics
        text, comp, dur = asr_transcribe(model, audio_f32)
        metrics.compute_sec += comp
        metrics.audio_sec += len(audio_f32)/SAMPLE_RATE
        metrics.chunks += 1

        if tag == "partial":
            if text:
                print("[partial] " + text)
        else:
            if text:
                print("[final] " + text)

    print("[mic] Start speaking. Ctrl+C to stop.")

    try:
        with MicStream() as ms:
            while True:
                frame = ms.read()

                if frame is None:
                    continue

                if len(frame) < FRAME_SAMPLES:
                    continue
                frame = frame[:FRAME_SAMPLES]

                for tag, audio in vad.push_and_maybe_yield(frame):
                    transcription_q.put((audio, tag))
    except KeyboardInterrupt:
        pass
    finally:
        transcription_q.put((None, None))
        print("\n[main] Waiting for worker to finish...")
        worker_thread.join()
        rtf = metrics.compute_sec/max(metrics.audio_sec, 1e-6)
        print(f"chunks = {metrics.chunks}, rtf = {rtf}")

if __name__ == "__main__":
    main()









