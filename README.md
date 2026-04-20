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
├── experiments/demo/             # v2 pipeline — shot_made, rim tracker, FG stats
│   ├── run_demo.py               # v1 end-to-end runner (ball-disc rule)
│   ├── run_demo_v2.py            # v2 runner (rim-plane crossing + occlusion)
│   ├── shot_detector.py          # v1 shot rule
│   ├── shot_made_v2.py           # v2 state machine for made detection
│   ├── shot_attempt_v2.py        # v2 multi-predicate attempt rule
│   ├── rim_tracker.py            # hybrid static-anchor + cv2 tracker
│   ├── duo_tracker.py            # 2-slot persistent player identity
│   ├── evaluate_v2.py            # precision/recall/F1 vs GT csv
│   ├── offline_sweep_v2.py       # CPU param sweep over existing tracks
│   ├── aggregate_v2.py           # CSV + summary.md + charts
│   ├── ARCHITECTURE_v2.md        # v2 design doc
│   └── FINDINGS_v2.md            # v2 sweep results
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

## Shot detection v2 (experiments/demo)

A drop-in replacement for the main pipeline's shot rule lives under
[`experiments/demo/`](experiments/demo/). It addresses v1's three known failure
modes — dribble-triggered false-positive attempts, missed makes when the
ball is net-occluded for 5–15 frames, and static-hoop drift on
moving-camera clips — without retraining the detector.

```bash
# run v2 end-to-end on a 70 s window (GPU)
.venv/bin/python experiments/demo/run_demo_v2.py \
    --video 1v1-mk.mov --start 268 --end 338 \
    --hoop experiments/demo/hoop_configs/1v1-mk.json \
    --model yolov8l.pt --imgsz 640 \
    --out experiments/demo/outputs_v2_live/my_run

# offline CPU param sweep over existing tracks.jsonl files (fast iteration)
.venv/bin/python experiments/demo/offline_sweep_v2.py \
    --tracks-glob 'experiments/demo/outputs/run*.tracks.jsonl' \
    --matrix experiments/demo/matrix_v2_offline.csv \
    --outdir experiments/demo/outputs_v2

# aggregate N runs into csv + markdown + charts
.venv/bin/python experiments/demo/aggregate_v2.py \
    --outdir experiments/demo/analysis_v2

# evaluate a run against ground truth
.venv/bin/python experiments/demo/evaluate_v2.py \
    --pred experiments/demo/outputs_v2_live/my_run.events.json \
    --gt experiments/demo/gt/1v1-mk_t268-338.csv \
    --tolerance 15
```

On the latest sweep (4,860 replays across 15 tunable rows),
v2 cut `shot_attempt` FPs **−65%** and lifted `shot_made` recall
**+54%** vs v1. See [`experiments/demo/FINDINGS_v2.md`](experiments/demo/FINDINGS_v2.md)
for the numbers and [`experiments/demo/ARCHITECTURE_v2.md`](experiments/demo/ARCHITECTURE_v2.md)
for the design.

---

## Custom basketball checkpoint (optional)

Point the v2 runner at a YOLOv8 `.pt` that includes a `hoop`/`rim`
class to replace the drift-prone rim tracker with per-frame
detections:

```bash
# list the three supported providers
python experiments/demo/download_basketball_model.py --list

# fetch (you supply trust + credentials, nothing is auto-pulled)
python experiments/demo/download_basketball_model.py --provider local \
    --source /path/to/your/verified/best.pt \
    --target experiments/demo/weights/basketball.pt

# run v2 with the custom model; hoop detections feed the rim tracker
python experiments/demo/run_demo_v2.py \
    --video 1v1-mk.mov --start 268 --end 338 \
    --hoop experiments/demo/hoop_configs/1v1-mk.json \
    --custom-model experiments/demo/weights/basketball.pt \
    --hoop-conf 0.4 \
    --out experiments/demo/outputs_v2_live/mk_custom
```

Canonical class names the pipeline normalises to via
[custom_tracker.py](experiments/demo/custom_tracker.py):
`player`, `ball`, `hoop`, `backboard`. Aliases cover common variants
(`person`, `basketball`, `rim`, `basketball-hoop`, `net`, …). Add more
via the `class_aliases` kwarg or a future CLI flag.

Security note: `YOLO(path.pt)` deserializes pickle. Only load weights
you've verified — either from your own training run, a pinned
HuggingFace revision, or a Roboflow export tied to your API key. The
helper prints a sha256 hash after download for future integrity checks.

---

## Slurm batch submission (v2)

```bash
# offline CPU sweep (fast, ~2 min for 324 tracks × 15 configs)
sbatch experiments/demo/offline_sweep_v2.sbatch

# live GPU matrix (12 tasks: 3 videos × rim-tracker variants)
sbatch --array=0-11 experiments/demo/run_matrix_v2_live.sbatch

# aggregator, runs afterany the above finish
sbatch --dependency=afterany:<offline_id>:<live_id> \
       experiments/demo/aggregate_v2.sbatch
```

---

## License

[MIT](LICENSE).
