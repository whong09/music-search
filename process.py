import os
import re
import csv
import json
import queue
import logging
import threading
from datetime import datetime
from time import perf_counter
from pathlib import Path
from multiprocessing import Process, Manager, cpu_count, JoinableQueue
from logging.handlers import RotatingFileHandler

import boto3
import numpy as np
from boto3.s3.transfer import TransferConfig, S3Transfer
from filelock import FileLock

import utils  # your utils.py with download_mp3, get_top_video_url, batch_transcribe, midi_to_chunks

#── CONFIG ─────────────────────────────────────────────────────────────────────
LOCAL = Path("pipeline")
# ensure directories
LOCAL.mkdir(exist_ok=True)
AUDIO_DIR = LOCAL / "audio"
MIDI_DIR = LOCAL / "midi"
DATA_DIR = LOCAL / "data"
for d in (AUDIO_DIR, MIDI_DIR, DATA_DIR): d.mkdir(parents=True, exist_ok=True)

# input files
SONGS_FILE = Path("songs.txt")
SONGS_LEFT_FILE = Path("songs-left.txt")
# prepare songs-left
if SONGS_LEFT_FILE.exists():
    src = SONGS_LEFT_FILE
else:
    src = SONGS_FILE
    if SONGS_FILE.exists():
        SONGS_LEFT_FILE.write_bytes(SONGS_FILE.read_bytes())
# load songs_left list
SONGS_LEFT = []
with open(src, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) >= 2:
            SONGS_LEFT.append((row[0], row[1]))
if not SONGS_LEFT:
    logger = logging.getLogger()
    logger.error("No songs found in songs-left.txt or songs.txt")

# AWS / batching
BUCKET = "midi2vec"
REGION = "us-west-1"
CPU_BATCH_SIZE = 8
UPLOAD_BATCH_SIZE = 16
CPU_WORKERS = 4
IO_WORKERS = 8
UPLOAD_WORKERS = 2

# checkpoint files (jsonl)
DOWNLOAD_FILE = LOCAL / ".downloaded.jsonl"
PROCESSED_FILE = LOCAL / ".processed.jsonl"
UPLOADED_FILE  = LOCAL / ".uploaded.jsonl"
DOWNLOAD_LOCK = str(DOWNLOAD_FILE) + ".lock"
PROCESSED_LOCK = str(PROCESSED_FILE) + ".lock"
UPLOADED_LOCK  = str(UPLOADED_FILE) + ".lock"

#── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = LOCAL / ".pipeline.log"
handler = logging.handlers.RotatingFileHandler(
    filename=str(LOG_FILE), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
formatter = logging.Formatter("%(asctime)s [%(processName)s|%(threadName)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# AWS S3 setup
s3_client = boto3.client("s3", region_name=REGION)
transfer_config = TransferConfig(
    multipart_threshold=8*1024*1024,
    max_concurrency=UPLOAD_WORKERS,
    multipart_chunksize=8*1024*1024,
    use_threads=True
)
s3_transfer = S3Transfer(s3_client, config=transfer_config)

# Locks
dl_lock = FileLock(DOWNLOAD_LOCK)
pr_lock = FileLock(PROCESSED_LOCK)
up_lock = FileLock(UPLOADED_LOCK)

#── Checkpoint helpers ─────────────────────────────────────────────────────────
def load_jsonl(path, lock):
    recs = []
    with lock:
        if path.exists():
            for line in path.open('r', encoding='utf-8'):
                try:
                    recs.append(json.loads(line))
                except Exception:
                    logger.exception(f"Failed to parse JSONL line in {path}")
    return recs


def append_jsonl(path, lock, record):
    with lock:
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

#── Utils ─────────────────────────────────────────────────────────────────────
def sanitize(s): return re.sub(r'-+', '-', re.sub(r'[^0-9a-zA-Z]+','-', s)).strip('-').lower()

#── Pipeline components ───────────────────────────────────────────────────────
def io_worker(download_q, mp3_q, downloaded_set):
    while True:
        try:
            artist, title = download_q.get(timeout=1)
        except queue.Empty:
            return
        try:
            key = f"{sanitize(artist)}_{sanitize(title)}"
            if key in downloaded_set:
                logger.info(f"[IO] skip {key}")
                download_q.task_done()
                continue

            url = utils.quick_scrape_top_url(f"{artist} {title} Piano Cover")
            if not url:
                logger.warning(f"[IO] no result {key}")
                download_q.task_done()
                continue

            logger.info(f"[IO] download {key} from {url}")
            mp3 = AUDIO_DIR / f"{key}.mp3"
            utils.download_mp3(key, url, output_dir=str(AUDIO_DIR))

            rec = {'key': key, 'url': url, 'ts': datetime.utcnow().isoformat() + 'Z'}
            append_jsonl(DOWNLOAD_FILE, dl_lock, rec)
            downloaded_set.add(key)
            mp3_q.put({**rec, 'mp3': str(mp3), 'start': perf_counter()})
        except Exception:
            logger.exception(f"[IO] unexpected error processing {artist}-{title}")
        finally:
            try:
                download_q.task_done()
            except Exception:
                pass


def cpu_worker(mp3_q, upload_q, processed_set, stop_evt):
    buf = []
    while not (stop_evt.is_set() and mp3_q.empty()):
        try:
            item = mp3_q.get(timeout=1)
        except queue.Empty:
            continue

        key = item['key']
        npz_path = DATA_DIR / f"{key}.npz"

        if npz_path.exists():
            logger.info(f"[CPU] cached chunk -> upload {key}")
            upload_q.put({'key' : key, 'url' : item['url'], 'mid' : str(MIDI_DIR / f"{key}.mid"), 'npz' : str(npz_path), 'ts'  : item['ts'], })
            processed_set.add(key)
            mp3_q.task_done()
            continue

        buf.append(item)
        mp3_q.task_done()

        if len(buf) >= CPU_BATCH_SIZE:
            keys = [x['key'] for x in buf]
            try:
                _process_batch(buf, upload_q, processed_set)
            except Exception:
                logger.exception(f"[CPU] error processing batch: {keys}")
            finally:
                buf.clear()
    if buf:
        keys = [x['key'] for x in buf]
        try:
            _process_batch(buf, upload_q, processed_set)
        except Exception:
            logger.exception(f"[CPU] error processing final batch: {keys}")
        finally:
            buf.clear()


def _process_batch(batch, upload_q, processed_set):
    try:
        paths = [b['mp3'] for b in batch]
        utils.batch_transcribe(paths, str(MIDI_DIR))
        logger.info(f"[CPU] processing batch of {len(batch)}")
        for b in batch:
            key = b['key']
            mid = MIDI_DIR / f"{key}.mid"
            chunks = utils.midi_to_chunks(str(mid))
            npz = DATA_DIR / f"{key}.npz"
            ts = datetime.utcnow().isoformat() + 'Z'
            np.savez_compressed(str(npz), chunks=chunks)
            upload_q.put({'key': key, 'url': b['url'], 'mid': str(mid), 'npz': str(npz), 'ts': ts})
            logger.info(f'wrote to q {key}')
            append_jsonl(PROCESSED_FILE, pr_lock, {'key': key, 'url': b['url'], 'ts': ts})
        processed_set.add(key)
    except Exception:
        logger.exception(f"[CPU] failed inside _process_batch for {[b['key'] for b in batch]}")
        raise


def upload_worker(upload_q, uploaded_set, stop_evt):
    buf = []
    while not (stop_evt.is_set() and upload_q.empty()):
        try:
            item = upload_q.get(timeout=1)
        except queue.Empty:
            logger.info('queue empty?')
            continue

        key = item['key']
        if key in uploaded_set:
            continue

        buf.append(item)

        logger.info(key)
        logger.info(len(buf))

        if len(buf) >= UPLOAD_BATCH_SIZE:
            keys = [b['key'] for b in buf]
            try:
                _upload_batch(buf, uploaded_set)
            except Exception:
                logger.exception(f"[UPLOAD] error uploading batch: {keys}")
            finally:
                buf.clear()
    if buf:
        keys = [b['key'] for b in buf]
        try:
            _upload_batch(buf, uploaded_set)
        except Exception:
            logger.exception(f"[UPLOAD] error uploading final batch: {keys}")
        finally:
            buf.clear()


def _upload_batch(batch, uploaded_set):
    logger.info('attempting to upload')
    for b in batch:
        key = b['key']
        paths = {
            'mp3': AUDIO_DIR / f"{key}.mp3",
            'mid': MIDI_DIR   / f"{key}.mid",
            'npz': DATA_DIR   / f"{key}.npz",
        }
        for f, pref in (('mp3', 'audio'), ('mid', 'midi'), ('npz', 'data')):
            path = paths[f]
            try:
                s3_transfer.upload_file(path, BUCKET, f"{pref}/{key}.{f}")
                Path(path).unlink(missing_ok=True)
            except Exception:
                logger.exception(f"[UPLOAD] failed to upload {pref}/{key}.{f}")
                continue
        append_jsonl(UPLOADED_FILE, up_lock, {'key': key, 'url': b.get('url'), 'ts': datetime.utcnow().isoformat() + 'Z'})
        uploaded_set.add(key)
        logger.info(f"[DONE] {key}")

#── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # load checkpoints
    downloaded = load_jsonl(DOWNLOAD_FILE,  dl_lock)
    downloaded_set = {r['key'] for r in downloaded}
    downloaded_set |= {p.stem for p in AUDIO_DIR.glob("*.mp3")}

    processed = load_jsonl(PROCESSED_FILE,  pr_lock)
    processed_set  = {r['key'] for r in processed}
    processed_set |= {p.stem for p in DATA_DIR.glob("*.npz")}

    uploaded_set   = {r['key'] for r in load_jsonl(UPLOADED_FILE, up_lock)}

    print(downloaded_set)
    print(processed_set)
    print(uploaded_set)

    # remove processed from songs_left
    remaining = [s for s in SONGS_LEFT if f"{sanitize(s[0])}_{sanitize(s[1])}" not in uploaded_set]
    with open(SONGS_LEFT_FILE, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t'); w.writerows(remaining)

    # clear uploaded checkpoint
    with pr_lock:
        UPLOADED_FILE.write_text('')

    remaining_keys = {f"{sanitize(artist)}_{sanitize(title)}" for artist, title in remaining}
    remaining_downloaded = [r for r in downloaded if r['key'] in remaining_keys]
    remaining_processed = [r for r in processed if r['key'] in remaining_keys]
    
    with DOWNLOAD_FILE.open('w', encoding='utf-8') as f:
        for r in remaining_downloaded:
            f.write(json.dumps(r) + '\n')

    with DOWNLOAD_FILE.open('w', encoding='utf-8') as f:
        for r in remaining_downloaded:
            f.write(json.dumps(r) + '\n')

    # queues and event
    manager = Manager(); 
    mp3_q = JoinableQueue(); 
    upload_q = JoinableQueue();
    download_q = queue.Queue(); 
    stop_evt = manager.Event();

    # enqueue to download
    for artist, title in remaining:
        key = f"{sanitize(artist)}_{sanitize(title)}"
        if key not in downloaded_set:
            download_q.put((artist, title))
    # enqueue to process
    for key in downloaded_set - processed_set:
        mp3_q.put({'key': key, 'url': None, 'mp3': str(AUDIO_DIR / f"{key}.mp3"), 'ts' : datetime.utcnow().isoformat()+'Z'})
    # enqueue to upload
    for key in processed_set - uploaded_set:
        upload_q.put({'key': key, 'url': None, 'mid': str(MIDI_DIR / f"{key}.mid"), 'npz': str(DATA_DIR / f"{key}.npz"), 'mp3': str(AUDIO_DIR / f"{key}.mp3"),'ts' : datetime.utcnow().isoformat()+'Z'})

    # start workers
    ios = []; procs = []; ups = []
    for _ in range(IO_WORKERS):
        t = threading.Thread(target=io_worker, args=(download_q, mp3_q, downloaded_set), daemon=True)
        t.start(); ios.append(t)
    for _ in range(CPU_WORKERS):
        p = Process(target=cpu_worker, args=(mp3_q, upload_q, processed_set, stop_evt), daemon=True)
        p.start(); procs.append(p)
    for _ in range(UPLOAD_WORKERS):
        p = Process(target=upload_worker, args=(upload_q, uploaded_set, stop_evt), daemon=True)
        p.start(); ups.append(p)

    # wait
    download_q.join()
    for t in ios: t.join()
    stop_evt.set()
    for p in procs: p.join()
    mp3_q.join()
    for t in ups: t.join()
    logger.info("Pipeline complete")
