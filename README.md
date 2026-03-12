# CourtVision

A neuro-symbolic basketball analytics system that turns raw 1v1 street-basketball footage into a timestamped event log and a natural-language scouting summary.

CourtVision separates visual perception from game logic on purpose: YOLOv8 + ByteTrack produce per-frame tracks, a small deterministic rule engine turns those tracks into possession and shot-attempt events, and a lightweight LLM (Gemini, with a local fallback) summarises the event log. Every intermediate artifact is serialisable, so when something goes wrong you can point at which stage is broken.

This project was built for **CS 4501: Computer Vision** at the University of Virginia.

---

## Quick start

```bash
# 1. clone and enter the repo
git clone https://github.com/<your-org>/courtvision.git
cd courtvision

# 2. create a venv (python 3.11+ required by ultralytics / torch 2)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. install python deps
pip install -r requirements.txt

# 4. (optional) set your gemini key for richer narratives
echo 'GEMINI_API_KEY=your-key-here' > .env

# 5. run on a short slice first
python main.py --video 1v1-mk.mov --start 0 --end 60
```

The first run downloads `yolov8m.pt` into the repo root (~50 MB) via Ultralytics. Subsequent runs reuse the cached weights.

Outputs land in `outputs/stats_report.txt` (path configurable).

---

## Configuration

All knobs live in [`config.yaml`](config.yaml). CLI flags override file values; environment variables (via the slurm script) override CLI flags.
