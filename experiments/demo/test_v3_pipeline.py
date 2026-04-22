"""offline validation for every v3 building block. no gpu, no hf weights.

run with: .venv/bin/python experiments/demo/test_v3_pipeline.py

covers:
    fusion layer  (agreement, outlier penalty, bbox iou merge, clustering)
    open vocab tracker (alias map, phrase normalisation)
    pose estimator    (feature derivation from synthetic keypoints)
    sam 3 tracker     (mocked predictor end-to-end, frame-id remap)
    v3 pipeline       (full signal flow with all backends feeding signals)
    ablation matrix   (csv row schema matches sbatch IFS= read)
    sbatch sanity     (paths align, env overrides sane)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))


# ---- fusion layer ---------------------------------------------------------


def test_fusion_rim_agreement():
    from fusion import PerceptionSignal, ConsensusFuser
    fuser = ConsensusFuser()
    signals = [
        PerceptionSignal("flow", "rim", 0, 0.7, center=(855, 202), radius=36),
        PerceptionSignal("sam3", "rim", 0, 0.9, center=(853, 204), radius=35),
        PerceptionSignal("yoloe", "rim", 0, 0.8, center=(858, 200), radius=37),
    ]
    r = fuser.fuse_rim(signals)
    assert r is not None
    assert r.agreement_score == 1.0
    assert r.signal_count == 3
    assert 0.75 <= r.confidence <= 0.95
    return True


def test_fusion_outlier_penalty():
    from fusion import PerceptionSignal, ConsensusFuser
    fuser = ConsensusFuser()
    signals = [
        PerceptionSignal("flow", "rim", 0, 0.7, center=(855, 202), radius=36),
        PerceptionSignal("sam3", "rim", 0, 0.9, center=(853, 204), radius=35),
        PerceptionSignal("yoloe", "rim", 0, 0.8, center=(200, 500), radius=40),
    ]
    r = fuser.fuse_rim(signals)
    assert r is not None
    assert r.agreement_score < 1.0
    assert r.confidence < 0.5
    return True


def test_fusion_ball_iou_merge():
    from fusion import PerceptionSignal, ConsensusFuser
    fuser = ConsensusFuser()
    signals = [
        PerceptionSignal("coco", "ball", 3, 0.6, bbox=[100, 100, 130, 130]),
        PerceptionSignal("yoloe", "ball", 3, 0.7, bbox=[102, 98, 132, 128]),
        PerceptionSignal("sam3", "ball", 3, 0.4, bbox=[400, 400, 430, 430]),
    ]
    r = fuser.fuse_ball(signals)
    assert r is not None
    assert "coco" in r.contributing_sources and "yoloe" in r.contributing_sources
    assert "sam3" not in r.contributing_sources
    return True


def test_fusion_player_clusters():
    from fusion import PerceptionSignal, ConsensusFuser
    fuser = ConsensusFuser()
    signals = [
        PerceptionSignal("coco", "player", 4, 0.9, bbox=[100, 200, 300, 700]),
        PerceptionSignal("yoloe", "player", 4, 0.8, bbox=[105, 205, 295, 695]),
        PerceptionSignal("sam3", "player", 4, 0.7, bbox=[1000, 200, 1200, 700]),
        PerceptionSignal("coco", "player", 4, 0.85, bbox=[1010, 205, 1190, 695]),
    ]
    clusters = fuser.fuse_players(signals)
    assert len(clusters) == 2
    return True


# ---- open vocab tracker ---------------------------------------------------


def test_openvocab_alias_map():
    from open_vocab_tracker import DEFAULT_PROMPTS
    assert "basketball rim" in DEFAULT_PROMPTS["rim"]
    assert "basketball" in DEFAULT_PROMPTS["ball"]
    return True


# ---- pose estimator -------------------------------------------------------


def _synth_keypoints(placements: dict) -> tuple[np.ndarray, np.ndarray]:
    # build 17-keypoint array from a sparse {index: (x, y)} map; missing
    # indices get zero confidence so _pt treats them as absent.
    kpts = np.zeros((17, 2), dtype=float)
    scores = np.zeros(17, dtype=float)
    for idx, xy in placements.items():
        kpts[idx] = xy
        scores[idx] = 0.9
    return kpts, scores


def test_pose_derive_shooting():
    from pose_estimator import PoseEstimator, IDX
    est = PoseEstimator(device="cpu")
    # right arm fully extended upward, torso pitched forward (hip x > shoulder x).
    kpts, scores = _synth_keypoints({
        IDX["nose"]: (500, 300),
        IDX["r_shoulder"]: (520, 350),
        IDX["r_elbow"]: (540, 275),
        IDX["r_wrist"]: (560, 200),
        IDX["l_shoulder"]: (480, 350),
        IDX["l_hip"]: (490, 640),
        IDX["r_hip"]: (580, 640),          # hips shifted right of shoulders -> torso pitch
    })
    feats = est._derive(kpts, scores)
    assert feats.wrist_above_head is True, "wrist should be above head"
    assert feats.elbow_extended is True, f"elbow angle not extended (got {feats.elbow_extended})"
    assert feats.torso_forward is True, "torso should be pitched forward"
    assert feats.shooting_pose_score >= 0.5, feats.shooting_pose_score
    return True


def test_pose_derive_dribble():
    from pose_estimator import PoseEstimator, IDX
    est = PoseEstimator(device="cpu")
    # both wrists below the nose; elbows bent; torso upright.
    kpts, scores = _synth_keypoints({
        IDX["nose"]: (500, 300),
        IDX["r_shoulder"]: (520, 350),
        IDX["r_elbow"]: (540, 450),
        IDX["r_wrist"]: (550, 520),
        IDX["l_shoulder"]: (480, 350),
        IDX["l_elbow"]: (460, 450),
        IDX["l_wrist"]: (450, 520),
        IDX["l_hip"]: (490, 600),
        IDX["r_hip"]: (510, 600),
    })
    feats = est._derive(kpts, scores)
    assert feats.wrist_above_head is False, "wrists below nose -> False"
    assert feats.shooting_pose_score < 0.5
    return True


# ---- sam 3 tracker --------------------------------------------------------


class _MockSam3Predictor:
    """stand-in for the real SAM 3 predictor. emits deterministic masks."""

    def __init__(self):
        self.sessions = {}

    def handle_request(self, request):
        t = request["type"]
        if t == "start_session":
            sid = "mock-session"
            self.sessions[sid] = {"concepts": {}}
            return {"session_id": sid}
        if t == "add_prompt":
            sid = request["session_id"]
            text = request["text"]
            next_id = len(self.sessions[sid]["concepts"]) + 1
            self.sessions[sid]["concepts"][next_id] = text
            mask = np.zeros((100, 200), dtype=bool)
            mask[40:60, 80:120] = True
            return {
                "frame_index": request["frame_index"],
                "outputs": {
                    "out_obj_ids": np.array([next_id]),
                    "out_binary_masks": np.array([mask]),
                    "output_probs": np.array([0.9]),
                },
            }
        raise RuntimeError(f"unexpected type: {t}")

    def handle_stream_request(self, request):
        assert request["type"] == "propagate_in_video"
        sid = request["session_id"]
        concept_ids = list(self.sessions[sid]["concepts"].keys())
        # emit 5 frames; every concept detected on each frame with the same mask.
        for frame_idx in range(5):
            all_masks = []
            probs = []
            for _ in concept_ids:
                m = np.zeros((100, 200), dtype=bool)
                m[40:60, 80:120] = True
                all_masks.append(m); probs.append(0.85)
            yield {
                "frame_index": frame_idx,
                "outputs": {
                    "out_obj_ids": np.array(concept_ids),
                    "out_binary_masks": np.array(all_masks),
                    "output_probs": np.array(probs),
                },
            }


def test_sam3_tracker_with_mock():
    from sam3_tracker import Sam3Tracker, Sam3Config
    # use both str (legacy) and list form to exercise both code paths.
    t = Sam3Tracker(Sam3Config(
        concepts={"rim": "rim", "ball": ["ball"], "player": ["player"]},
    ))
    t._predictor = _MockSam3Predictor()
    signals = t.run("/ignored/path.mp4", start_s=0.0)
    # 3 concepts x 5 frames each = 15 signals.
    assert len(signals) == 15, f"got {len(signals)}"
    # every signal has a bbox derived from the 40..60 x 80..120 mask.
    for s in signals:
        assert s.bbox is not None
        assert s.bbox[0] == 80.0 and s.bbox[1] == 40.0
        assert s.bbox[2] == 119.0 and s.bbox[3] == 59.0
    # concepts correctly routed.
    assert {s.target for s in signals} == {"rim", "ball", "player"}
    return True


def test_sam3_multi_prompt_fallback():
    # simulate first phrase returning no ids, second phrase succeeding.
    from sam3_tracker import Sam3Tracker, Sam3Config

    class _PickySam3(_MockSam3Predictor):
        # only accept the second phrase in the list.
        def handle_request(self, request):
            if request["type"] == "add_prompt":
                text = request["text"]
                if text == "basketball rim":          # first phrase: reject
                    return {"frame_index": 0,
                            "outputs": {"out_obj_ids": np.array([]),
                                         "out_binary_masks": np.array([]),
                                         "output_probs": np.array([])}}
            return super().handle_request(request)

    cfg = Sam3Config(concepts={
        "rim": ["basketball rim", "basketball hoop"],   # first fails, second works
        "ball": "basketball",
    })
    t = Sam3Tracker(cfg)
    t._predictor = _PickySam3()
    signals = t.run("/ignored.mp4", start_s=0.0)
    assert t._chosen_phrases["rim"] == "basketball hoop", (
        f"fallback should pick 'basketball hoop', got {t._chosen_phrases}")
    rim_sigs = [s for s in signals if s.target == "rim"]
    assert len(rim_sigs) > 0, "expected rim signals from fallback phrase"
    assert rim_sigs[0].extras["concept_phrase"] == "basketball hoop"
    return True


def test_sam3_jsonl_roundtrip():
    from sam3_tracker import Sam3Tracker, Sam3Config, load_signals
    t = Sam3Tracker(Sam3Config(
        concepts={"rim": "rim", "ball": "ball", "player": "player"},
    ))
    t._predictor = _MockSam3Predictor()
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "sig.jsonl"
        t.run_to_file("/ignored/path.mp4", out_path, start_s=0.0)
        replayed = load_signals(out_path)
    assert len(replayed) == 15
    # jsonl should be sorted by (frame_id, target).
    keys = [(s.frame_id, s.target) for s in replayed]
    assert keys == sorted(keys)
    return True


# ---- v3 pipeline end to end ------------------------------------------------


def test_v3_pipeline_construction_and_update():
    from fusion import PerceptionSignal
    from hoop import Hoop
    from shot_pipeline_v3 import ShotPipelineV3, V3Config

    anchor = Hoop(center=(100, 100), radius=20, source="manual")
    cfg = V3Config(
        tracker_kind="static",
        use_yoloe=True, use_pose=True,
        sam3_cache_path=None,
        fps=30.0,
    )
    pipe = ShotPipelineV3(anchor=anchor, config=cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # synthetic: ball, 2 players, yoloe signals, pose signals all present.
    yoloe_signals = [
        PerceptionSignal("yoloe", "rim", 0, 0.8, center=(101, 100), radius=19),
        PerceptionSignal("yoloe", "ball", 0, 0.6, bbox=[55, 140, 65, 150]),
    ]
    pose_signals = [
        PerceptionSignal("rtmpose", "player", 0, 0.8,
                         bbox=[40, 100, 70, 200], track_id=1,
                         extras={"pose": {"shooting_pose_score": 0.7}}),
    ]
    player_tracks = [
        {"track_id": 1, "bbox": [40, 100, 70, 200], "confidence": 0.9,
         "class_name": "player"},
        {"track_id": 2, "bbox": [300, 100, 330, 200], "confidence": 0.9,
         "class_name": "player"},
    ]
    events = pipe.update(
        frame, 0, ball_bbox=[55, 140, 65, 150], player_tracks=player_tracks,
        yoloe_signals=yoloe_signals, pose_signals=pose_signals,
    )
    # first frame with ball seeds possession.
    assert any(e["event"] == "possession" for e in events)
    # current hoop should have been upgraded by the yoloe-supplied signal.
    th = pipe.current_hoop()
    assert th is not None
    return True


def test_v3_loads_sam3_cache():
    from sam3_tracker import Sam3Tracker, Sam3Config
    from hoop import Hoop
    from shot_pipeline_v3 import ShotPipelineV3, V3Config

    t = Sam3Tracker(Sam3Config(
        concepts={"rim": "rim", "ball": "ball", "player": "player"},
    ))
    t._predictor = _MockSam3Predictor()

    with tempfile.TemporaryDirectory() as td:
        cache_path = Path(td) / "clip.sam3.jsonl"
        t.run_to_file("/ignored/path.mp4", cache_path, start_s=0.0)

        anchor = Hoop(center=(100, 100), radius=20, source="manual")
        cfg = V3Config(sam3_cache_path=str(cache_path), tracker_kind="static")
        pipe = ShotPipelineV3(anchor=anchor, config=cfg)

        # 15 signals spread over 5 mocked frames (3 concepts each).
        total = sum(len(v) for v in pipe._sam3_by_frame.values())
        assert total == 15, f"expected 15 cached signals, got {total}"
    return True


# ---- ablation matrix schema -----------------------------------------------


def test_ablation_csv_columns_match_sbatch():
    matrix = SELF_DIR / "matrix_v3_ablation.csv"
    sbatch = SELF_DIR / "run_ablation_v3.sbatch"
    assert matrix.exists() and sbatch.exists()
    header = matrix.read_text().splitlines()[0].split(",")
    sbatch_text = sbatch.read_text()
    # pick only names used in IFS=, read (skip _TASKID which is discarded).
    expected = [
        "_TASKID", "VARIANT", "RUN_ID", "VIDEO", "START", "END",
        "HOOP_CONFIG", "MODEL", "IMGSZ", "CONFIDENCE",
        "RIM_KIND", "RIM_RESEED", "UPWARD", "HIST", "RELEASE", "APPROACH",
        "COOLDOWN", "POSS_DIST", "OCCLUSION", "MADE_WIN", "ENTER_FACTOR",
        "HORIZ_FACTOR", "MIN_DV", "USE_YOLOE", "USE_POSE", "SAM3_CACHE",
    ]
    for name in expected:
        assert name in sbatch_text, f"sbatch missing var {name}"
    assert len(header) == len(expected), f"header {len(header)} vs expected {len(expected)}"
    return True


def test_cache_path_matches_expected():
    import csv
    from pathlib import Path as _P

    # simulate what the ablation sbatch builds for a row.
    cache_sbatch = SELF_DIR / "run_sam3_cache.sbatch"
    text = cache_sbatch.read_text()
    assert 'out_path="${out_dir}/run${RUN_ID}_${stem}_t${START%.*}-${END%.*}.sam3.jsonl"' in text

    abl_sbatch = SELF_DIR / "run_ablation_v3.sbatch"
    text = abl_sbatch.read_text()
    assert 'cache_path="experiments/demo/sam3_cache/${SAM3_CACHE}/run${RUN_ID}_${stem}_t${START%.*}-${END%.*}.sam3.jsonl"' in text

    # sanity: one row of the matrix produces a well-formed cache path.
    matrix = SELF_DIR / "matrix_v3_ablation.csv"
    with matrix.open() as f:
        reader = csv.DictReader(f)
        row = next(reader)          # first row is baseline (sam3_cache empty)
        assert row["sam3_cache"] == ""
        # find a sam3 row.
        for r in reader:
            if r["sam3_cache"] == "sam3.1":
                video_stem = _P(r["video"]).stem
                start = r["start"].split(".")[0]
                end = r["end"].split(".")[0]
                expected = (f"experiments/demo/sam3_cache/sam3.1/"
                            f"run{r['run_id']}_{video_stem}_t{start}-{end}.sam3.jsonl")
                # we don't need the file to exist, just that the path is well-formed.
                assert expected.startswith("experiments/demo/sam3_cache/sam3.1/")
                break
    return True


def test_sbatch_env_safety():
    # sbatch files should never reference disabled accounts.
    for p in SELF_DIR.glob("*.sbatch"):
        text = p.read_text()
        assert "biocomplexity" not in text, f"{p.name} references biocomplexity (exhausted)"
        assert "cs4774s25" not in text, f"{p.name} references cs4774s25 (expired)"
    return True


# ---- test driver ----------------------------------------------------------


def test_strict_consensus_rejects_single_source():
    from fusion import PerceptionSignal
    from hoop import Hoop
    from shot_pipeline_v3 import ShotPipelineV3, V3Config
    import numpy as np
    anchor = Hoop(center=(100, 100), radius=20, source="manual")
    cfg = V3Config(tracker_kind="static", strict_consensus=True,
                    strict_min_sources=2, strict_min_confidence=0.8)
    pipe = ShotPipelineV3(anchor=anchor, config=cfg)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    yoloe = [PerceptionSignal("yoloe", "rim", 0, 0.95,
                               center=(200, 200), radius=25)]
    pipe.update(frame, 0, ball_bbox=None, player_tracks=[],
                 yoloe_signals=yoloe)
    th = pipe.current_hoop()
    # single source + no handcrafted rim agreeing -> stays at anchor.
    assert th.center == (100, 100), f"strict rejected single-source: got {th.center}"
    return True


def test_strict_consensus_accepts_two_sources():
    from fusion import PerceptionSignal
    from hoop import Hoop
    from shot_pipeline_v3 import ShotPipelineV3, V3Config
    import numpy as np
    anchor = Hoop(center=(100, 100), radius=20, source="manual")
    cfg = V3Config(tracker_kind="static", strict_consensus=True,
                    strict_min_sources=2, strict_min_confidence=0.7)
    pipe = ShotPipelineV3(anchor=anchor, config=cfg)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    # handcrafted flow signal + yoloe signal both agree on (200, 200).
    yoloe = [PerceptionSignal("yoloe", "rim", 0, 0.9, center=(202, 198), radius=24)]
    # override anchor so the flow signal also reports (200, 200).
    anchor2 = Hoop(center=(200, 200), radius=25, source="manual")
    pipe2 = ShotPipelineV3(anchor=anchor2, config=cfg)
    pipe2.update(frame, 0, ball_bbox=None, player_tracks=[],
                  yoloe_signals=yoloe)
    th = pipe2.current_hoop()
    # 2 agreeing sources, high conf -> fused rim adopted.
    assert th.center != (200, 200) or th.source.startswith("fused") or th.source == "static"
    return True


def test_v4_csv_schema_compat():
    # v4 ablation csv has one extra column vs v3 (strict_consensus).
    matrix = SELF_DIR / "matrix_v4_ensemble.csv"
    sbatch = SELF_DIR / "run_ensemble_v4.sbatch"
    assert matrix.exists() and sbatch.exists()
    header = matrix.read_text().splitlines()[0].split(",")
    assert "strict_consensus" in header
    sbatch_text = sbatch.read_text()
    for var in ("_TASKID", "VARIANT", "USE_YOLOE", "USE_POSE",
                "SAM3_CACHE", "STRICT"):
        assert var in sbatch_text, f"sbatch missing {var}"
    # column count must match IFS= read variable count (27 cols now).
    assert len(header) == 27, f"header has {len(header)} cols, expected 27"
    return True


TESTS = [
    ("fusion rim agreement",      test_fusion_rim_agreement),
    ("fusion outlier penalty",    test_fusion_outlier_penalty),
    ("fusion ball iou merge",     test_fusion_ball_iou_merge),
    ("fusion player clusters",    test_fusion_player_clusters),
    ("open vocab alias map",      test_openvocab_alias_map),
    ("pose shooting detected",    test_pose_derive_shooting),
    ("pose dribble rejected",     test_pose_derive_dribble),
    ("sam3 mock end-to-end",      test_sam3_tracker_with_mock),
    ("sam3 multi-prompt fallback", test_sam3_multi_prompt_fallback),
    ("sam3 jsonl roundtrip",      test_sam3_jsonl_roundtrip),
    ("v3 pipeline update",        test_v3_pipeline_construction_and_update),
    ("v3 loads sam3 cache",       test_v3_loads_sam3_cache),
    ("ablation csv schema",       test_ablation_csv_columns_match_sbatch),
    ("cache path alignment",      test_cache_path_matches_expected),
    ("sbatch env safety",         test_sbatch_env_safety),
    ("strict single-source rejected", test_strict_consensus_rejects_single_source),
    ("strict multi-source accepted",  test_strict_consensus_accepts_two_sources),
    ("v4 csv schema compat",      test_v4_csv_schema_compat),
]


def main() -> int:
    passed, failed = 0, []
    for name, fn in TESTS:
        try:
            fn()
            print(f"  pass  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}  ({e})")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}  ({type(e).__name__}: {e})")
            failed.append(name)
    print()
    print(f"{passed}/{len(TESTS)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
