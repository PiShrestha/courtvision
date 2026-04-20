# Per-clip report (v2)

Focus is strictly the two primary slots (P1, P2). Any tracks that escalated past those ids are bucketed as 'other' — if that bucket is non-empty the DuoTracker lost identity during the clip.

total clips: **12**

## headline table

| clip | window | dur | fps | P1 att/made/FG% | P2 att/made/FG% | overall FG% | P1 poss | P2 poss | other |
|---|---|---|---|---|---|---|---|---|---|
| 1v1-ddg | 929-991s | 1m01s | 60.0 | 0/0/— | 0/0/— | — | 0.0s | 0.0s | 0 events from ids none |
| 1v1-mk | 268-338s | 1m09s | 29.969 | 11/0/0% | 0/0/— | 0% | 1m07s | 1.6s | 0 events from ids none |
| 1v1-ddg | 1017-1078s | 1m01s | 60.0 | 0/0/— | 0/0/— | — | 56.1s | 0.0s | 0 events from ids none |
| 1v1-mk | 374-450s | 1m16s | 29.969 | 10/0/0% | 0/0/— | 0% | 1m13s | 0.0s | 0 events from ids none |
| 1v1-ddg | 929-991s | 1m01s | 60.0 | 0/0/— | 0/0/— | — | 23.8s | 0.0s | 0 events from ids none |
| 1v1-ddg | 1017-1078s | 1m01s | 60.0 | 0/0/— | 0/0/— | — | 0.0s | 0.0s | 0 events from ids none |
| 1v1-jason | 254-317s | 1m02s | 60.0 | 4/0/0% | 3/1/33% | 14% | 27.8s | 33.9s | 0 events from ids none |
| 1v1-jason | 651-724s | 1m12s | 60.0 | 7/1/14% | 6/0/0% | 8% | 44.4s | 28.3s | 0 events from ids none |
| 1v1-nasir | 789-852s | 1m03s | 59.94 | 3/1/33% | 1/0/0% | 25% | 26.1s | 26.7s | 0 events from ids none |
| 1v1-nasir | 1773-1852s | 1m18s | 59.94 | 5/0/0% | 0/0/— | 0% | 1m02s | 16.6s | 0 events from ids none |
| 1v1-roy | 179-268s | 1m29s | 60.0 | 0/0/— | 4/0/0% | 0% | 7.6s | 35.3s | 0 events from ids none |
| 1v1-roy | 321-399s | 1m17s | 60.0 | 3/1/33% | 10/1/10% | 15% | 19.4s | 57.9s | 0 events from ids none |

---

## 1v1-ddg  (929–991s)

- **duration:** 1m01s at 60.0 fps (3668 frames processed)
- **model:** yolov8x.pt @ imgsz=1280
- **hoop:** {'center': [960, 200], 'radius': 40, 'source': 'manual_guess'}
- **total events logged:** 0

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | — | 0.0s |
| P2 | 0 | 0 | 0 | — | 0.0s |
| **total** | **0** | **0** | 0 | **—** | 0.0s (0.0 coverage) |

---

## 1v1-mk  (268–338s)

- **duration:** 1m09s at 29.969 fps (2089 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [855, 202], 'radius': 36, 'source': 'auto'}
- **total events logged:** 25

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 11 | 0 | 10 | 0% | 1m07s |
| P2 | 0 | 0 | 0 | — | 1.6s |
| **total** | **11** | **0** | 10 | **0%** | 1m09s (0.99 coverage) |

---

## 1v1-ddg  (1017–1078s)

- **duration:** 1m01s at 60.0 fps (3704 frames processed)
- **model:** yolov8x.pt @ imgsz=1280
- **hoop:** {'center': [960, 200], 'radius': 40, 'source': 'manual_guess'}
- **total events logged:** 1

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | — | 56.1s |
| P2 | 0 | 0 | 0 | — | 0.0s |
| **total** | **0** | **0** | 0 | **—** | 56.1s (0.91 coverage) |

---

## 1v1-mk  (374–450s)

- **duration:** 1m16s at 29.969 fps (2280 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [855, 202], 'radius': 36, 'source': 'auto'}
- **total events logged:** 21

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 10 | 0 | 10 | 0% | 1m13s |
| P2 | 0 | 0 | 0 | — | 0.0s |
| **total** | **10** | **0** | 10 | **0%** | 1m13s (0.96 coverage) |

---

## 1v1-ddg  (929–991s)

- **duration:** 1m01s at 60.0 fps (3668 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [960, 200], 'radius': 40, 'source': 'manual_guess'}
- **total events logged:** 1

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | — | 23.8s |
| P2 | 0 | 0 | 0 | — | 0.0s |
| **total** | **0** | **0** | 0 | **—** | 23.8s (0.39 coverage) |

---

## 1v1-ddg  (1017–1078s)

- **duration:** 1m01s at 60.0 fps (3704 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [960, 200], 'radius': 40, 'source': 'manual_guess'}
- **total events logged:** 0

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | — | 0.0s |
| P2 | 0 | 0 | 0 | — | 0.0s |
| **total** | **0** | **0** | 0 | **—** | 0.0s (0.0 coverage) |

---

## 1v1-jason  (254–317s)

- **duration:** 1m02s at 60.0 fps (3726 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [960, 200], 'radius': 40, 'source': 'manual_guess'}
- **total events logged:** 26

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 4 | 0 | 3 | 0% | 27.8s |
| P2 | 3 | 1 | 2 | 33% | 33.9s |
| **total** | **7** | **1** | 5 | **14%** | 1m01s (0.99 coverage) |

---

## 1v1-jason  (651–724s)

- **duration:** 1m12s at 60.0 fps (4364 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [960, 200], 'radius': 40, 'source': 'manual_guess'}
- **total events logged:** 47

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 7 | 1 | 6 | 14% | 44.4s |
| P2 | 6 | 0 | 6 | 0% | 28.3s |
| **total** | **13** | **1** | 12 | **8%** | 1m12s (1.0 coverage) |

---

## 1v1-nasir  (789–852s)

- **duration:** 1m03s at 59.94 fps (3819 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [962, 256], 'radius': 46, 'source': 'auto_candidate'}
- **total events logged:** 21

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 3 | 1 | 2 | 33% | 26.1s |
| P2 | 1 | 0 | 1 | 0% | 26.7s |
| **total** | **4** | **1** | 3 | **25%** | 52.8s (0.83 coverage) |

---

## 1v1-nasir  (1773–1852s)

- **duration:** 1m18s at 59.94 fps (4725 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [962, 256], 'radius': 46, 'source': 'auto_candidate'}
- **total events logged:** 17

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 5 | 0 | 5 | 0% | 1m02s |
| P2 | 0 | 0 | 0 | — | 16.6s |
| **total** | **5** | **0** | 5 | **0%** | 1m18s (1.0 coverage) |

---

## 1v1-roy  (179–268s)

- **duration:** 1m29s at 60.0 fps (5357 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [1412, 477], 'radius': 56, 'source': 'auto_candidate'}
- **total events logged:** 12

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | — | 7.6s |
| P2 | 4 | 0 | 4 | 0% | 35.3s |
| **total** | **4** | **0** | 4 | **0%** | 42.9s (0.48 coverage) |

---

## 1v1-roy  (321–399s)

- **duration:** 1m17s at 60.0 fps (4640 frames processed)
- **model:** yolov8l.pt @ imgsz=640
- **hoop:** {'center': [1412, 477], 'radius': 56, 'source': 'auto_candidate'}
- **total events logged:** 51

### per-player metrics

| player | attempts | made | miss | FG% | possession |
|---|---:|---:|---:|---:|---:|
| P1 | 3 | 1 | 2 | 33% | 19.4s |
| P2 | 10 | 1 | 8 | 10% | 57.9s |
| **total** | **13** | **2** | 10 | **15%** | 1m17s (1.0 coverage) |
