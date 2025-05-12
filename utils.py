from __future__ import annotations

import os
import requests

import pretty_midi
import numpy as np
from scipy.ndimage import zoom

from pathlib import Path
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

from yt_dlp import YoutubeDL
import re, requests, urllib.parse

import functools, shelve, threading

import time

_CACHE_PATH = Path("pipeline/.url_cache")
_cache_lock = threading.Lock()

def disk_cache(func):
    """
    Decorator: memoize `func(arg)` persistently in a shelve DB.
    Thread‑safe, process‑safe (shelve uses anydbm).
    """
    @functools.wraps(func)
    def wrapper(arg, *a, **kw):
        with _cache_lock, shelve.open(str(_CACHE_PATH)) as db:
            if arg in db:
                return db[arg]
        # not cached → compute outside the lock
        result = func(arg, *a, **kw)
        with _cache_lock, shelve.open(str(_CACHE_PATH)) as db:
            db[arg] = result
        return result
    return wrapper

def rate_limit(calls_per_sec: float):
    """
    Decorator: limit *total* call rate across all threads/processes.
    Uses a token bucket in a file‑lock.
    """
    bucket_lock = threading.Lock()
    capacity = calls_per_sec
    tokens = capacity
    last = time.perf_counter()

    def decorator(func):
        def wrapped(*args, **kw):
            nonlocal tokens, last
            while True:
                with bucket_lock:
                    now = time.perf_counter()
                    # add tokens based on elapsed time
                    tokens = min(capacity, tokens + (now - last) * calls_per_sec)
                    last = now
                    if tokens >= 1:
                        tokens -= 1
                        break
                time.sleep(0.01)   # wait 10 ms
            return func(*args, **kw)
        return wrapped
    return decorator

model = Model(ICASSP_2022_MODEL_PATH)

_YT_ID_RE = re.compile(r'\"videoId\":\"([A-Za-z0-9_-]{11})\"')

@disk_cache
@rate_limit(8)
def quick_scrape_top_url(query: str, timeout: float = 5.0):
    """
    Very‑lightweight search scrape: 1 request, 1 regex, returns first /watch v ID.
    Zero Data‑API quota.  Returns None if no video found.
    """
    q = urllib.parse.quote_plus(query)
    url = f"https://www.youtube.com/results?search_query={q}&sp=EgIQAQ%253D%253D"  # filter: type=video
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.7",
    }

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()          # will raise on 403/429 so you can back‑off

    m = _YT_ID_RE.search(r.text)
    return f"https://youtu.be/{m.group(1)}" if m else None

class QuotaExceeded(Exception):
    """Daily YouTube API quota exhausted (wait until midnight PT)."""

@disk_cache
@rate_limit(8)
def get_top_video_url(query, api_key=None, max_results=10, retries=5):
    api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not set")

    params = {
        "key": api_key,
        "part": "id",
        "type": "video",
        "maxResults": max_results,
        "q": query,
        "order": "relevance",
        "fields": "items(id/videoId)",
    }

    backoff = 1
    for _ in range(retries):
        r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
        if r.status_code == 200:
            items = r.json().get("items", [])
            return f"https://youtu.be/{items[0]['id']['videoId']}" if items else None

        try:
            reason = r.json()["error"]["errors"][0]["reason"]
        except Exception:
            reason = "unknown"

        if reason in ("rateLimitExceeded", "userRateLimitExceeded"):
            time.sleep(backoff)
            backoff = min(backoff * 2, 32)   # exponential back‑off, max 32 s
            continue

        if reason == "quotaExceeded":
            raise QuotaExceeded("YouTube daily quota exhausted")

        r.raise_for_status()

    return None

def get_active_note_bounds(pm: pretty_midi.PrettyMIDI, buffer_s=0.25):
    """Returns (start_time, end_time) covering all note events with buffer."""
    note_times = [
        (note.start, note.end)
        for instrument in pm.instruments if not instrument.is_drum
        for note in instrument.notes
    ]
    if not note_times:
        return 0.0, pm.get_end_time()

    note_starts, note_ends = zip(*note_times)
    start = max(0.0, min(note_starts) - buffer_s)
    end = min(pm.get_end_time(), max(note_ends) + buffer_s)
    return start, end

def resample_chunk(chunk, target_len=512):
    """Resize time axis of (128, T) piano roll to (128, target_len)."""
    current_len = chunk.shape[1]
    if current_len == target_len:
        return chunk
    zoom_factor = target_len / current_len
    return zoom(chunk, (1, zoom_factor), order=1).astype(np.float32)

def midi_to_chunks(midi_path, chunks_per_song=8, fs=4, target_len=512):
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    # Trim silence
    t_start, t_end = get_active_note_bounds(pm)
    trimmed = pm.get_piano_roll(fs=fs, times=np.arange(t_start, t_end, 1/fs))

    # Normalize
    roll = np.clip(trimmed / 127.0, 0.0, 1.0)
    total_frames = roll.shape[1]
    
    chunk_width = total_frames // chunks_per_song
    chunks = []

    for i in range(chunks_per_song):
        start = i * chunk_width
        end = total_frames if i == chunks_per_song - 1 else (i + 1) * chunk_width
        chunk = roll[:, start:end]
        chunk = resample_chunk(chunk, target_len)
        chunks.append(chunk)

    return chunks

def analyze(chunks, threshold=1e-3):
    for i, c in enumerate(chunks):
        density = np.count_nonzero(c) / c.size
        activity = np.sum(c)
        active_pitches = np.count_nonzero(np.sum(c, axis=1))
        active_frames = np.count_nonzero(np.sum(c, axis=0))

        # Use a meaningful threshold to avoid floating point noise
        nonzero_vals = c[c > threshold]
        min_nonzero = nonzero_vals.min() if nonzero_vals.size > 0 else 0.0

        print(
            f"Chunk {i}: shape={c.shape}, "
            f"max={c.max():.2f}, min_nonzero={min_nonzero:.3f}, "
            f"density={density:.4f}, total={activity:.2f}, "
            f"pitches={active_pitches}, frames={active_frames}"
        )

def batch_transcribe(
    audio_paths: list[str],
    output_dir: str,
    onset_thresh: float = 0.5,
    frame_thresh: float = 0.7,
    min_note_len: float = 0.1,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for src in audio_paths:
        model_out, midi_data, note_events = predict(
            src,
            model,
            onset_threshold=onset_thresh,
            frame_threshold=frame_thresh,
            minimum_note_length=min_note_len,
        )

        stem = Path(src).stem
        dst  = Path(output_dir) / f"{stem}.mid"
        midi_data.write(str(dst))

def download_mp3(name: str,
                 url: str,
                 abr: int = 64,
                 sample_rate: int = 22050,
                 channels: int = 1,
                 output_dir: str = '.'):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': f'worstaudio[abr>={abr}]/bestaudio',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': str(abr),
        }],
        'postprocessor_args': [
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-b:a', f'{abr}k'
        ],
        'outtmpl': os.path.join(output_dir, f'{name}.%(ext)s'),
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
