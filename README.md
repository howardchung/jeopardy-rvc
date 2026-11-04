# jeopardy-rvc

Standalone HTTP service for Jeopardy! RVC v2 + Edge-TTS AI voice generation using `AlexTrebek.zip.part1` and `AlexTrebek.zip.part2`.

## Features
- Automatically selects GPU (`cuda`, `mps`) or `cpu` and determines the optimal worker pool size.
- Priority queue (`priority=true` on-demand synthesis jumps ahead of background pre-generation).
- Automatic request coalescing and client-disconnect cancellation.
- Returns MP3 audio bytes over HTTP so `jeopardy` and `jeopardy-rvc` can run on separate hosts.

## Usage
```bash
pip install -r requirements.txt
python3 server.py
```

### Environment Variables
- `HOST`: Bind address (default `0.0.0.0`)
- `PORT`: Bind port (default `8084`)
- `RVC_NUM_WORKERS`: Optional override for worker process count
- `RVC_DEVICE`: Optional override for PyTorch device (e.g. `cuda:0`, `mps`, `cpu`)
