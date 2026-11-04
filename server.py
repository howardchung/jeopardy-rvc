#!/usr/bin/env python3
"""
Standalone HTTP server for Jeopardy! RVC v2 + Edge-TTS AI voice generation.
Determines the optimal number of worker processes automatically based on
available GPUs (CUDA / MPS) or CPU cores, manages a priority queue of
synthesis jobs, caches generated MP3 files in ./audio, and serves them directly over HTTP.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from rvc_worker import select_device_and_workers

ROOT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = ROOT_DIR / "audio"
WORKER_SCRIPT = ROOT_DIR / "rvc_worker.py"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DEVICE, MAX_WORKERS = select_device_and_workers()


def sanitize_hash(raw_hash: str | None, text: str = "") -> str:
    if raw_hash:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", os.path.basename(str(raw_hash)))
        if cleaned:
            return cleaned
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class SynthesisJob:
    job_id: str
    key: str
    text: str
    output_path: Path
    priority: bool
    seq: int
    future: asyncio.Future[Path]
    waiters: int = 1
    started: bool = False
    cancelled: bool = False


@dataclass
class WorkerProcess:
    worker_id: int
    proc: asyncio.subprocess.Process
    ready: bool = False
    device: str = DEFAULT_DEVICE
    active_job: SynthesisJob | None = field(default=None)


class RVCWorkerPool:
    def __init__(self, max_workers: int):
        self.max_workers = max(1, max_workers)
        self.workers: list[WorkerProcess] = []
        self.pending_jobs: list[SynthesisJob] = []
        self.inflight_by_key: dict[str, SynthesisJob] = {}
        self._seq = 0
        self._worker_counter = 0
        self._spawning_workers = 0
        self._spawn_lock = asyncio.Lock()

    def _sort_queue(self):
        self.pending_jobs.sort(key=lambda j: (0 if j.priority else 1, j.seq))

    async def _spawn_worker(self) -> WorkerProcess:
        worker_idx = self._worker_counter
        self._worker_counter += 1
        worker_id = worker_idx + 1

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["RVC_THREADS_PER_WORKER"] = "1"
        env["RVC_WORKER_INDEX"] = str(worker_idx)

        self._spawning_workers += 1
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(WORKER_SCRIPT),
                cwd=str(ROOT_DIR),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,
                env=env,
            )
            slot = WorkerProcess(worker_id=worker_id, proc=proc)
            self.workers.append(slot)
        finally:
            self._spawning_workers -= 1

        asyncio.create_task(self._read_worker_stdout(slot))
        asyncio.create_task(self._watch_worker_exit(slot))
        return slot

    async def _read_worker_stdout(self, slot: WorkerProcess):
        assert slot.proc.stdout is not None
        while True:
            line_bytes = await slot.proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception as e:
                print(
                    f"[jeopardy-rvc] Failed to parse worker #{slot.worker_id} stdout: {line} ({e})",
                    file=sys.stderr,
                )
                continue

            msg_type = msg.get("type")
            if msg_type == "ready":
                slot.ready = True
                slot.device = msg.get("device", DEFAULT_DEVICE)
                print(
                    f"[jeopardy-rvc] Worker #{slot.worker_id}/{self.max_workers} ready on {slot.device}",
                    file=sys.stderr,
                )
                await self._dispatch()
                continue

            if msg_type == "init_error":
                err = RuntimeError(msg.get("error", "Worker initialization failed"))
                print(
                    f"[jeopardy-rvc] Worker #{slot.worker_id} init error: {err}",
                    file=sys.stderr,
                )
                if slot.active_job and not slot.active_job.future.done():
                    slot.active_job.future.set_exception(err)
                slot.active_job = None
                continue

            job = slot.active_job
            if job and msg.get("id") == job.job_id:
                slot.active_job = None
                if msg.get("ok"):
                    if not job.future.done():
                        job.future.set_result(job.output_path)
                else:
                    err = RuntimeError(msg.get("error", "Synthesis failed"))
                    print(
                        f"[jeopardy-rvc] Worker #{slot.worker_id} synthesis error: {err}",
                        file=sys.stderr,
                    )
                    if not job.future.done():
                        job.future.set_exception(err)
                await self._dispatch()

    async def _watch_worker_exit(self, slot: WorkerProcess):
        code = await slot.proc.wait()
        print(
            f"[jeopardy-rvc] Worker #{slot.worker_id} exited with code {code}",
            file=sys.stderr,
        )
        if slot in self.workers:
            self.workers.remove(slot)
        if slot.active_job:
            job = slot.active_job
            slot.active_job = None
            if not job.future.done():
                job.future.set_exception(
                    RuntimeError(f"Worker #{slot.worker_id} exited with code {code}")
                )
        if self.pending_jobs:
            await asyncio.sleep(0.5)
            await self._dispatch()

    async def _ensure_workers(self):
        async with self._spawn_lock:
            while True:
                busy = sum(1 for w in self.workers if w.active_job is not None)
                active_pending = sum(
                    1 for j in self.pending_jobs if not j.cancelled
                )
                desired = min(self.max_workers, busy + active_pending)
                if len(self.workers) + self._spawning_workers >= desired:
                    break
                await self._spawn_worker()

    async def _dispatch(self):
        self.pending_jobs = [j for j in self.pending_jobs if not j.cancelled]
        await self._ensure_workers()

        while self.pending_jobs:
            front = self.pending_jobs[0]
            if front.cancelled:
                self.pending_jobs.pop(0)
                continue

            if front.output_path.is_file() and front.output_path.stat().st_size > 0:
                self.pending_jobs.pop(0)
                if not front.future.done():
                    front.future.set_result(front.output_path)
                continue

            idle_worker = next(
                (w for w in self.workers if w.ready and w.active_job is None), None
            )
            if idle_worker is None:
                break

            job = self.pending_jobs.pop(0)
            job.started = True
            idle_worker.active_job = job
            payload = (
                json.dumps(
                    {
                        "id": job.job_id,
                        "text": job.text,
                        "output_path": str(job.output_path),
                    }
                )
                + "\n"
            )
            assert idle_worker.proc.stdin is not None
            idle_worker.proc.stdin.write(payload.encode("utf-8"))
            await idle_worker.proc.stdin.drain()

    async def synthesize(
        self, text: str, raw_hash: str | None = None, priority: bool = False
    ) -> Path:
        key = sanitize_hash(raw_hash, text)
        output_path = AUDIO_DIR / f"{key}.mp3"

        if output_path.is_file() and output_path.stat().st_size > 0:
            return output_path

        existing = self.inflight_by_key.get(key)
        if existing is not None and not existing.cancelled:
            existing.waiters += 1
            if priority and not existing.priority and not existing.started:
                existing.priority = True
                self._sort_queue()
            try:
                return await asyncio.shield(existing.future)
            except asyncio.CancelledError:
                existing.waiters -= 1
                if existing.waiters <= 0 and not existing.started:
                    existing.cancelled = True
                    self.inflight_by_key.pop(key, None)
                raise

        loop = asyncio.get_running_loop()
        self._seq += 1
        job = SynthesisJob(
            job_id=f"job_{self._seq}_{key}",
            key=key,
            text=text,
            output_path=output_path,
            priority=priority,
            seq=self._seq,
            future=loop.create_future(),
        )
        self.inflight_by_key[key] = job
        self.pending_jobs.append(job)
        self._sort_queue()
        await self._dispatch()

        try:
            return await asyncio.shield(job.future)
        except asyncio.CancelledError:
            job.waiters -= 1
            if job.waiters <= 0 and not job.started:
                job.cancelled = True
                self.inflight_by_key.pop(key, None)
            raise
        finally:
            if job.future.done():
                self.inflight_by_key.pop(key, None)


pool = RVCWorkerPool(MAX_WORKERS)


async def send_http_response(
    writer: asyncio.StreamWriter,
    status: int,
    reason: str,
    content_type: str,
    body: bytes,
    extra_headers: dict[str, str] | None = None,
    head_only: bool = False,
):
    hdrs = [
        f"HTTP/1.1 {status} {reason}",
        f"Content-Type: {content_type}",
        f"Content-Length: {len(body)}",
        "Access-Control-Allow-Origin: *",
        "Access-Control-Allow-Methods: GET, POST, HEAD, OPTIONS",
        "Access-Control-Allow-Headers: Content-Type, Range",
        "Connection: close",
    ]
    if extra_headers:
        for k, v in extra_headers.items():
            hdrs.append(f"{k}: {v}")
    header_bytes = ("\r\n".join(hdrs) + "\r\n\r\n").encode("latin-1")
    writer.write(header_bytes if head_only else header_bytes + body)
    await writer.drain()


def extract_audio_hash_from_path(path: str) -> str | None:
    for prefix in ("/audio/", "/aivoice/", "/gradio_api/file=audio/output/"):
        if path.startswith(prefix):
            filename = path[len(prefix) :]
            if filename.lower().endswith(".mp3"):
                filename = filename[:-4]
            return sanitize_hash(filename)
    return None


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
        if not request_line:
            return
        parts = request_line.decode("latin-1", errors="replace").strip().split(" ")
        if len(parts) < 2:
            return
        method, raw_target = parts[0].upper(), parts[1]

        headers: dict[str, str] = {}
        while True:
            hline = await asyncio.wait_for(reader.readline(), timeout=30.0)
            if not hline or hline in (b"\r\n", b"\n"):
                break
            decoded = hline.decode("latin-1", errors="replace").strip()
            if ":" in decoded:
                k, v = decoded.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        content_length = int(headers.get("content-length", "0"))
        body_bytes = (
            await asyncio.wait_for(reader.readexactly(content_length), timeout=30.0)
            if content_length > 0
            else b""
        )

        parsed = urllib.parse.urlsplit(raw_target)
        path = urllib.parse.unquote(parsed.path)
        query = urllib.parse.parse_qs(parsed.query)

        if method == "OPTIONS":
            await send_http_response(writer, 204, "No Content", "text/plain", b"")
            return

        if path in ("/ping", "/health") and method in ("GET", "HEAD"):
            payload = json.dumps(
                {
                    "status": "ok",
                    "default_device": DEFAULT_DEVICE,
                    "max_workers": pool.max_workers,
                    "spawned_workers": len(pool.workers),
                    "ready_workers": sum(1 for w in pool.workers if w.ready),
                    "busy_workers": sum(
                        1 for w in pool.workers if w.active_job is not None
                    ),
                    "queued_jobs": len(pool.pending_jobs),
                }
            ).encode("utf-8")
            await send_http_response(
                writer,
                200,
                "OK",
                "application/json",
                payload,
                head_only=(method == "HEAD"),
            )
            return

        # Serve MP3 files directly from the Python RVC server
        audio_hash = extract_audio_hash_from_path(path)
        if audio_hash and method in ("GET", "HEAD"):
            file_path = AUDIO_DIR / f"{audio_hash}.mp3"
            text_param = (query.get("text", [""])[0] or "").strip()
            if (not file_path.is_file() or file_path.stat().st_size == 0) and text_param:
                synth_task = asyncio.create_task(
                    pool.synthesize(
                        text=text_param, raw_hash=audio_hash, priority=True
                    )
                )
                disconnect_task = asyncio.create_task(reader.read(1))
                done, _ = await asyncio.wait(
                    {synth_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in done and not synth_task.done():
                    synth_task.cancel()
                    try:
                        await synth_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    return
                disconnect_task.cancel()
                file_path = await synth_task

            if file_path.is_file() and file_path.stat().st_size > 0:
                mp3_bytes = file_path.read_bytes()
                await send_http_response(
                    writer,
                    200,
                    "OK",
                    "audio/mpeg",
                    mp3_bytes,
                    extra_headers={
                        "Cache-Control": "public, max-age=31536000, immutable",
                        "Accept-Ranges": "none",
                    },
                    head_only=(method == "HEAD"),
                )
            else:
                await send_http_response(
                    writer,
                    404,
                    "Not Found",
                    "application/json",
                    json.dumps({"error": "Audio not found"}).encode("utf-8"),
                    head_only=(method == "HEAD"),
                )
            return

        if path == "/synthesize" and method in ("GET", "POST"):
            text = ""
            raw_hash: str | None = None
            priority = False

            if method == "POST" and body_bytes:
                data = json.loads(body_bytes.decode("utf-8"))
                text = str(data.get("text") or "").strip()
                raw_hash = data.get("hash")
                priority = bool(data.get("priority", False))
            else:
                text = (query.get("text", [""])[0] or "").strip()
                raw_hash = query.get("hash", [None])[0]
                priority = query.get("priority", ["false"])[0].lower() in (
                    "1",
                    "true",
                    "yes",
                )

            if not text or len(text) > 10000:
                err_body = json.dumps({"error": "Invalid or missing text"}).encode(
                    "utf-8"
                )
                await send_http_response(
                    writer, 400, "Bad Request", "application/json", err_body
                )
                return

            key = sanitize_hash(raw_hash, text)
            synth_task = asyncio.create_task(
                pool.synthesize(text=text, raw_hash=key, priority=priority)
            )
            disconnect_task = asyncio.create_task(reader.read(1))

            done, _ = await asyncio.wait(
                {synth_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if disconnect_task in done and not synth_task.done():
                synth_task.cancel()
                try:
                    await synth_task
                except (asyncio.CancelledError, Exception):
                    pass
                return

            disconnect_task.cancel()
            await synth_task
            url_path = f"/audio/{key}.mp3"
            resp_payload = json.dumps(
                {"ok": True, "hash": key, "url": url_path}
            ).encode("utf-8")
            await send_http_response(
                writer, 200, "OK", "application/json", resp_payload
            )
            return

        await send_http_response(
            writer,
            404,
            "Not Found",
            "application/json",
            json.dumps({"error": "Not found"}).encode("utf-8"),
            head_only=(method == "HEAD"),
        )
    except Exception as e:
        print(f"[jeopardy-rvc] Request error: {e}", file=sys.stderr)
        try:
            await send_http_response(
                writer,
                500,
                "Internal Server Error",
                "application/json",
                json.dumps({"error": str(e)}).encode("utf-8"),
            )
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8084"))
    server = await asyncio.start_server(handle_client, host, port)
    print(
        f"[jeopardy-rvc] Listening on http://{host}:{port} "
        f"(device={DEFAULT_DEVICE}, max_workers={pool.max_workers})",
        file=sys.stderr,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
