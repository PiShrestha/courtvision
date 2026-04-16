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
│   └── event_log.py              # ordered, serialisable event list
│
├── narrative/                    # stage 3: events -> scouting summary
│   └── generator.py              # gemini api or deterministic fallback
│
├── scripts/
│   ├── run_courtvision.sh        # slurm + bash runner, env-var driven
│   ├── sweep_90s.sh              # submit the 6-config detector sweep
│   └── compare_sweep.sh          # side-by-side summary across sweep outputs
│
├── presentation/                 # slide deck + matplotlib charts
├── requirements.txt
├── README.md
├── ARCHITECTURE.md               # stage-by-stage design
└── CONSTRAINTS.md                # what works, what doesn't, tradeoffs
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the data-flow diagram and stage contracts, and [CONSTRAINTS.md](CONSTRAINTS.md) for an honest discussion of what the system can and cannot do.

---

## Running on UVA HPC (Slurm)

The repo includes a single parameterised Slurm script. It works both as `sbatch` and as plain `bash`.

```bash
# submit a 90-second slice to the gpu queue
VIDEO=1v1-mk.mov START=0 END=90 sbatch scripts/run_courtvision.sh

# run a 6-config parameter sweep overnight
bash scripts/sweep_90s.sh

# tomorrow morning: see a side-by-side table of every sweep result
bash scripts/compare_sweep.sh
```

The default Slurm account is `cs6770_sp26`; edit `scripts/run_courtvision.sh` if yours differs.

Each config produces a report named `outputs/<video>_<model>_conf<NN>_imgsz<###>_<tracker>_<jobid>.txt` so sweep artifacts never collide.

---

## How the three stages fit together

1. **Perception** — YOLOv8 runs on every (or every Nth) frame. Ultralytics' ByteTrack assigns persistent track IDs and handles short occlusions via Kalman prediction. Output: `{frame_id, frame_height, frame_width, tracks: [...]}`.

2. **Symbolic reasoning** — a small Python rule engine assigns possession to the nearest player within a distance threshold and flags a shot attempt on sharp upward ball motion. Thresholds live in court coordinates when a homography is configured and fall back to auto-scaled pixel distances otherwise. Output: `[{frame_id, event, player}, ...]`.

3. **Narrative** — the event log is formatted into a prompt for Gemini (when `GEMINI_API_KEY` is set) or passed to a deterministic local summariser that tallies possessions / shot attempts per player.

A VLM never looks at raw frames in this design. It sees a pre-validated event stream, which is what stops the "ordered hallucination" failure mode that end-to-end video-to-text models tend to produce.

---

## What maps to what in the proposal

| Proposal component | Status | Code |
|---|---|---|
| YOLO detection + ByteTrack | deployed | [perception/tracker.py](perception/tracker.py) |
| Symbolic event rules (possession, shot_attempt) | deployed | [logic/rules.py](logic/rules.py) |
| Planar homography module | scaffolded; works when a static calibration is given | [logic/homography.py](logic/homography.py) |
| VLM narrative (Gemini, Llama 3.2-Vision planned) | deployed (Gemini) | [narrative/generator.py](narrative/generator.py) |
| DINOv2 / SAM 2 re-identification | deferred past midterm (explicit in proposal) | — |

Final-stage upgrades (per-frame court keypoint detection for moving-camera homography, custom basketball YOLO checkpoint with a hoop class, Llama 3.2-Vision) are listed in [CONSTRAINTS.md](CONSTRAINTS.md) § "Future work".

---

## License

[MIT](LICENSE).
