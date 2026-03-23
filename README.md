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

```yaml
video:
  start_seconds: 0
  end_seconds: null     # null = end of file
  stride: 1             # keep every nth frame

perception:
  model: yolov8m.pt     # n | s | m | l | x
  confidence: 0.35
  imgsz: 640
  tracker: bytetrack.yaml   # or botsort.yaml

symbolic:
  court_possession_dist: 6.0   # feet, used with homography
  pixel_possession_dist: 110   # pixels at 360p, auto-scaled

narrative:
  gemini_model: gemini-1.5-flash

homography_config: null   # path to json, or null
out_txt: outputs/stats_report.txt
```

Override from the command line:

```bash
python main.py --video clip.mov --model yolov8x.pt --imgsz 1280 --confidence 0.25
```

---

## Project layout

```
courtvision/
├── main.py                       # entry point, wires the three stages
├── config.py                     # loads config.yaml and applies cli overrides
├── config.yaml                   # all tunable knobs in one place
├── report.py                     # builds the text stats report
│
├── perception/                   # stage 1: video -> tracks
│   ├── pipeline.py               # frame loop, homography projection
│   └── tracker.py                # yolov8 + ultralytics bytetrack/botsort
│
├── logic/                        # stage 2: tracks -> events
│   ├── homography.py             # pixel <-> court coordinate mapping
│   ├── rules.py                  # possession + shot_attempt rules
