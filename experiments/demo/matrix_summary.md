# Demo Experiment Matrix

total runs: **324**
videos: 3 (1v1-ddg.mp4, 1v1-jason.mp4, 1v1-mk.mov)
windows per video: 3
detector configs per window: 12
shot-detector configs per detector: 3

## axes

### detector (full cross product)
- model: ['yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
- imgsz: [640, 1280]
- confidence: [0.25, 0.35]

### shot detector presets
- {'upward_trigger': -8.0, 'made_window': 60, 'hoop_pad': 20}
- {'upward_trigger': -12.0, 'made_window': 45, 'hoop_pad': 14}
- {'upward_trigger': -16.0, 'made_window': 30, 'hoop_pad': 8}

### fixed
- tracker: bytetrack.yaml
- possession_dist_px: 200.0